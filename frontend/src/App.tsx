// eslint-disable-next-line @typescript-eslint/no-unused-vars
import React, { useState, useCallback, useMemo } from "react";
// eslint-disable-next-line @typescript-eslint/no-unused-vars
import {
  AppBar,
  Toolbar,
  Typography,
  Container,
  Grid,
  Card,
  CardContent,
  TextField,
  Tabs,
  Tab,
  Button,
  LinearProgress,
  Paper,
  Table,
  TableHead,
  TableRow,
  TableCell,
  TableBody,
  Box,
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  Dialog,
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  DialogTitle,
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  DialogContent,
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  DialogActions,
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  CircularProgress,
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  ThemeProvider,
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  createTheme,
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  CssBaseline,
} from "@mui/material";

// eslint-disable-next-line @typescript-eslint/no-unused-vars
type TabKey = "current" | "thrust" | "rpm";
// eslint-disable-next-line @typescript-eslint/no-unused-vars
type MainTabKey = "test" | "history";

// eslint-disable-next-line @typescript-eslint/no-unused-vars
interface TaskState {
  file: File | null;
  taskId: string | null;
  progress: number;
  fps: number;
}

// eslint-disable-next-line @typescript-eslint/no-unused-vars
interface MetaState {
  prop: string;
  motor: string;
  esc: string;
  voltage: string;
}

// eslint-disable-next-line @typescript-eslint/no-unused-vars
interface TableRowData {
  [key: string]: string | number | null;
}

// eslint-disable-next-line @typescript-eslint/no-unused-vars
interface HistoryResult {
  id: number;
  session_id: string;
  prop_name: string;
  motor_name: string;
  esc_name: string;
  voltage: number | null;
  created_at: string;
  csv_path: string | null;
  graph_paths: string[];
  table_data: TableRowData[];
}

// eslint-disable-next-line @typescript-eslint/no-unused-vars
interface HistorySearchParams {
  prop: string | null;
  motor: string | null;
  esc: string | null;
}

// eslint-disable-next-line @typescript-eslint/no-unused-vars
const darkTheme = createTheme({
  palette: {
    mode: "dark",
    background: {
      default: "#0a0e27",
      paper: "#141829",
    },
    primary: {
      main: "#00d4ff",
      light: "#4dd0e1",
      dark: "#00838f",
    },
    secondary: {
      main: "#ff6b9d",
      light: "#ff8ab5",
      dark: "#c21857",
    },
    success: { main: "#4caf50" },
    error: { main: "#f44336" },
    text: { primary: "#e0e0e0", secondary: "#b0bec5" },
    divider: "#2c3e50",
  },
  typography: {
    fontFamily: '"Inter","Roboto","Helvetica","Arial",sans-serif',
    h5: { fontWeight: 600, letterSpacing: 0.5 },
    h6: { fontWeight: 600 },
    body2: { color: "#b0bec5" },
  },
});

// eslint-disable-next-line @typescript-eslint/no-unused-vars
export default function App(): JSX.Element {
  const [meta, setMeta] = useState<MetaState>({
    prop: "",
    motor: "",
    esc: "",
    voltage: "",
  });

  const [mainTab, setMainTab] = useState<MainTabKey>("test");
  const [tab, setTab] = useState<TabKey>("current");
  const [sessionId, setSessionId] = useState<string | null>(null);

  const [tasks, setTasks] = useState<Record<TabKey, TaskState>>({
    current: { file: null, taskId: null, progress: 0, fps: 5 },
    thrust: { file: null, taskId: null, progress: 0, fps: 5 },
    rpm: { file: null, taskId: null, progress: 0, fps: 5 },
  });

  const [tableRows, setTableRows] = useState<TableRowData[]>([]);
  const [graphs, setGraphs] = useState<string[]>([]);
  const [csvUrl, setCsvUrl] = useState<string | null>(null);

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [historyResults, setHistoryResults] = useState<HistoryResult[]>([]);
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [historyLoading, setHistoryLoading] = useState<boolean>(false);
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [historySearch, setHistorySearch] = useState<HistorySearchParams>({
    prop: null,
    motor: null,
    esc: null,
  });
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [selectedHistoryResult, setSelectedHistoryResult] =
    useState<HistoryResult | null>(null);
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [historyDetailOpen, setHistoryDetailOpen] = useState<boolean>(false);

  const backend = "http://127.0.0.1:8000";

  const handleMetaChange = useCallback((field: keyof MetaState, value: string) => {
    setMeta((prev) => ({ ...prev, [field]: value }));
  }, []);

  const task_id_encode = (taskId: string) => taskId;

  const pollTask = useCallback(
    (t: TabKey, taskId: string, sess: string): void => {
      const timer = setInterval(async () => {
        try {
          const res = await fetch(`${backend}/progress/${task_id_encode(taskId)}`);
          const p = await res.json();
          setTasks((prev) => ({
            ...prev,
            [t]: { ...prev[t], progress: p.progress ?? 0 },
          }));

          if (p.status === "done" || p.progress >= 100) {
            clearInterval(timer);
            const repRes = await fetch(`${backend}/session/${sess}/result`);
            if (repRes.ok) {
              const rep = await repRes.json();
              setTableRows(rep.table || []);
              setGraphs((rep.graphs || []).map((g: string) => `${backend}${g}`));
              setCsvUrl(rep.csv_url ? `${backend}${rep.csv_url}` : null);
            }
          }
        } catch {
          /* ignore */
        }
      }, 1500);
    },
    [backend]
  );

  const handleFilePick = useCallback((t: TabKey, f: File | null) => {
    setTasks((prev) => ({ ...prev, [t]: { ...prev[t], file: f } }));
  }, []);

  const handleFpsChange = useCallback((t: TabKey, value: number) => {
    setTasks((prev) => ({ ...prev, [t]: { ...prev[t], fps: value } }));
  }, []);

  const startUpload = useCallback(
    async (t: TabKey): Promise<void> => {
      const task = tasks[t];
      if (!task.file) {
        alert("Please choose a video file.");
        return;
      }
      const form = new FormData();
      form.append("file", task.file);
      form.append("video_type", t);
      if (sessionId) form.append("session_id", sessionId);
      form.append("prop", meta.prop);
      form.append("motor", meta.motor);
      form.append("esc", meta.esc);
      form.append("voltage", meta.voltage);
      form.append("fps", task.fps.toString());

      try {
        const res = await fetch(`${backend}/start`, { method: "POST", body: form });
        const data = await res.json();
        const newSession = data.session_id as string;
        const tId = data.task_id as string;
        setSessionId(newSession);
        setTasks((prev) => ({
          ...prev,
          [t]: { ...prev[t], taskId: tId, progress: 0 },
        }));
        pollTask(t, tId, newSession);
      } catch (error) {
        console.error("Upload error:", error);
        alert("Failed to upload video");
      }
    },
    [meta, sessionId, tasks, pollTask, backend]
  );

  const InputRow = useMemo(
    () => (
      <Grid container spacing={2} sx={{ mb: 2 }}>
        <Grid item xs={12} sm={3}>
          <TextField
            label="Prop"
            fullWidth
            value={meta.prop}
            onChange={(e) => handleMetaChange("prop", e.target.value)}
            variant="outlined"
          />
        </Grid>
        <Grid item xs={12} sm={3}>
          <TextField
            label="Motor"
            fullWidth
            value={meta.motor}
            onChange={(e) => handleMetaChange("motor", e.target.value)}
            variant="outlined"
          />
        </Grid>
        <Grid item xs={12} sm={3}>
          <TextField
            label="ESC"
            fullWidth
            value={meta.esc}
            onChange={(e) => handleMetaChange("esc", e.target.value)}
            variant="outlined"
          />
        </Grid>
        <Grid item xs={12} sm={3}>
          <TextField
            label="Voltage (V)"
            type="number"
            fullWidth
            value={meta.voltage}
            onChange={(e) => handleMetaChange("voltage", e.target.value)}
            variant="outlined"
          />
        </Grid>
      </Grid>
    ),
    [meta, handleMetaChange]
  );

  const TabPane = (t: TabKey, title: string): JSX.Element => {
    const st = tasks[t];
    return (
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            {title}
          </Typography>
          <Box sx={{ display: "flex", alignItems: "center", gap: 2 }}>
            <input
              type="file"
              accept="video/*"
              onChange={(e) => handleFilePick(t, e.target.files?.[0] || null)}
            />
            <Button variant="contained" onClick={() => startUpload(t)}>
              Upload & Start
            </Button>
            <TextField
              select
              label="FPS"
              value={st.fps}
              onChange={(e) => handleFpsChange(t, Number(e.target.value))}
              SelectProps={{ native: true }}
              variant="outlined"
              size="small"
              sx={{ width: 100 }}
            >
              {[...Array(10)].map((_, i) => (
                <option key={i + 1} value={i + 1}>
                  {i + 1}
                </option>
              ))}
              <option value={30}>30</option>
              <option value={60}>60</option>
              <option value={120}>120</option>
            </TextField>
          </Box>
          {st.taskId && (
            <Box sx={{ mt: 2 }}>
              <Typography variant="body2">Progress: {st.progress}%</Typography>
              <LinearProgress variant="determinate" value={st.progress} sx={{ mt: 1 }} />
            </Box>
          )}
        </CardContent>
      </Card>
    );
  };

  return (
    <ThemeProvider theme={darkTheme}>
      <CssBaseline />
      <AppBar
        position="static"
        sx={{
          background: "linear-gradient(135deg, #0a0e27 0%, #141829 100%)",
          borderBottom: "1px solid rgba(0, 212, 255, 0.2)",
        }}
      >
        <Toolbar>
          <Typography
            variant="h6"
            sx={{ flexGrow: 1, fontWeight: 700, letterSpacing: 1 }}
          >
            Propellor Test System
          </Typography>
          {sessionId && (
            <Typography variant="body2" sx={{ opacity: 0.8, fontFamily: "monospace" }}>
              Session: {sessionId.substring(0, 8)}...
            </Typography>
          )}
        </Toolbar>
      </AppBar>

      <Container sx={{ mt: 4, mb: 6 }}>
        <Tabs value={mainTab} onChange={(_e, v) => setMainTab(v)} sx={{ mb: 4 }}>
          <Tab label="Test" value="test" />
          <Tab label="History" value="history" />
        </Tabs>

        {mainTab === "test" && (
          <>
            <Card sx={{ mb: 3 }}>
              <CardContent>
                <Typography variant="h5" gutterBottom>
                  Test Setup
                </Typography>
                {InputRow}
                <Tabs value={tab} onChange={(_e, v) => setTab(v)} sx={{ mt: 2 }}>
                  <Tab label="Current Video" value="current" />
                  <Tab label="Thrust Video" value="thrust" />
                  <Tab label="RPM Video" value="rpm" />
                </Tabs>
                <Box sx={{ mt: 2 }}>
                  {tab === "current" && TabPane("current", "Upload Current Video")}
                  {tab === "thrust" && TabPane("thrust", "Upload Thrust Video")}
                  {tab === "rpm" && TabPane("rpm", "Upload RPM Video")}
                </Box>
              </CardContent>
            </Card>

            <Card>
              <CardContent>
                <Typography variant="h5" gutterBottom>
                  Test Data (Session Report)
                </Typography>
                {csvUrl && (
                  <Button variant="contained" href={csvUrl} sx={{ mb: 2 }}>
                    Download CSV
                  </Button>
                )}
                {tableRows.length > 0 ? (
                  <Paper sx={{ width: "100%", overflow: "auto", mb: 3 }}>
                    <Table size="small">
                      <TableHead>
                        <TableRow sx={{ backgroundColor: "rgba(0, 212, 255, 0.08)" }}>
                          <TableCell>Time (s)</TableCell>
                          <TableCell>Voltage (V)</TableCell>
                          <TableCell>Prop</TableCell>
                          <TableCell>Motor</TableCell>
                          <TableCell>ESC</TableCell>
                          <TableCell>Throttle</TableCell>
                          <TableCell>Current (A)</TableCell>
                          <TableCell>Power (W)</TableCell>
                          <TableCell>Thrust (G)</TableCell>
                          <TableCell>RPM</TableCell>
                          <TableCell>Efficiency (G/W)</TableCell>
                          <TableCell>Operating Temperature (°C)</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {tableRows.map((r, i) => (
                          <TableRow key={i}>
                            <TableCell>{r["Time (s)"]}</TableCell>
                            <TableCell>{r["Voltage (V)"]}</TableCell>
                            <TableCell>{r["Prop"]}</TableCell>
                            <TableCell>{r["Motor"]}</TableCell>
                            <TableCell>{r["ESC"]}</TableCell>
                            <TableCell>{r["Throttle"]}</TableCell>
                            <TableCell>{r["Current (A)"]}</TableCell>
                            <TableCell>{r["Power (W)"]}</TableCell>
                            <TableCell>{r["Thrust (G)"]}</TableCell>
                            <TableCell>{r["RPM"]}</TableCell>
                            <TableCell>{r["Efficiency (G/W)"]}</TableCell>
                            <TableCell>{r["Operating Temperature (°C)"]}</TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </Paper>
                ) : (
                  <Typography variant="body2" sx={{ opacity: 0.7 }}>
                    Upload and process your videos to see the merged table here.
                  </Typography>
                )}
                {graphs.length > 0 && (
                  <>
                    <Typography variant="h6" sx={{ mt: 3, mb: 2 }}>
                      Graphs
                    </Typography>
                    {graphs.map((g, i) => (
                      <Box key={i} sx={{ mt: 2 }}>
                        <img
                          src={g}
                          alt={`graph-${i}`}
                          style={{ width: "100%", borderRadius: 8 }}
                        />
                      </Box>
                    ))}
                  </>
                )}
              </CardContent>
            </Card>
          </>
        )}
      </Container>
    </ThemeProvider>
  );
}
