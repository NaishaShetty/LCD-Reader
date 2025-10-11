import React, { useState, useCallback, useMemo } from "react";
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
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  CircularProgress,
  ThemeProvider,
  createTheme,
  CssBaseline,
} from "@mui/material";

type TabKey = "current" | "thrust" | "rpm";
type MainTabKey = "test" | "history";

interface TaskState {
  file: File | null;
  taskId: string | null;
  progress: number;
}

interface MetaState {
  prop: string;
  motor: string;
  esc: string;
  voltage: string;
}

interface TableRowData {
  [key: string]: string | number | null;
}

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

interface HistorySearchParams {
  prop: string | null;
  motor: string | null;
  esc: string | null;
}

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

export default function App(): JSX.Element {
  const [meta, setMeta] = useState<MetaState>({
    prop: "",
    motor: "",
    esc: "",
    voltage: "",
  });

  // === ADDED: FPS state (default 5) ===
  const [fps, setFps] = useState<number>(5);

  const [mainTab, setMainTab] = useState<MainTabKey>("test");
  const [tab, setTab] = useState<TabKey>("current");
  const [sessionId, setSessionId] = useState<string | null>(null);

  const [tasks, setTasks] = useState<Record<TabKey, TaskState>>({
    current: { file: null, taskId: null, progress: 0 },
    thrust: { file: null, taskId: null, progress: 0 },
    rpm: { file: null, taskId: null, progress: 0 },
  });

  const [tableRows, setTableRows] = useState<TableRowData[]>([]);
  const [graphs, setGraphs] = useState<string[]>([]);
  const [csvUrl, setCsvUrl] = useState<string | null>(null);

  const [historyResults, setHistoryResults] = useState<HistoryResult[]>([]);
  const [historyLoading, setHistoryLoading] = useState<boolean>(false);
  const [historySearch, setHistorySearch] = useState<HistorySearchParams>({
    prop: null,
    motor: null,
    esc: null,
  });
  const [selectedHistoryResult, setSelectedHistoryResult] =
    useState<HistoryResult | null>(null);
  const [historyDetailOpen, setHistoryDetailOpen] = useState<boolean>(false);

  const backend = "http://127.0.0.1:8000";

  // Stable handlers
  const handleMetaChange = useCallback((field: keyof MetaState, value: string) => {
    setMeta((prev) => ({ ...prev, [field]: value }));
  }, []);

  const pollTask = useCallback(
    (t: TabKey, taskId: string, sess: string): void => {
      const timer = setInterval(async () => {
        try {
          const res = await fetch(`${backend}/progress/${task_id_encode(taskId)}`);
          const p = await res.json();
          setTasks((prev) => ({ ...prev, [t]: { ...prev[t], progress: p.progress ?? 0 } }));

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

  // helper to keep previous naming style if needed (no-op but keeps code stable)
  const task_id_encode = (taskId: string) => taskId;

  const fetchAllHistory = useCallback(async (): Promise<void> => {
    setHistoryLoading(true);
    try {
      const res = await fetch(`${backend}/history`);
      const data = await res.json();
      setHistoryResults(data.results || []);
    } finally {
      setHistoryLoading(false);
    }
  }, [backend]);

  const handleMainTabChange = useCallback(
    (_event: any, value: MainTabKey) => {
      setMainTab(value);
      if (value === "history") fetchAllHistory();
    },
    [fetchAllHistory]
  );

  const handleTabChange = useCallback((_event: any, value: TabKey) => setTab(value), []);

  const handleFilePick = useCallback((t: TabKey, f: File | null) => {
    setTasks((prev) => ({ ...prev, [t]: { ...prev[t], file: f } }));
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

      // === ADDED: send fps to backend ===
      form.append("fps", fps.toString());

      try {
        const res = await fetch(`${backend}/start`, { method: "POST", body: form });
        const data = await res.json();
        const newSession = data.session_id as string;
        const tId = data.task_id as string;
        setSessionId(newSession);
        setTasks((prev) => ({ ...prev, [t]: { ...prev[t], taskId: tId, progress: 0 } }));
        pollTask(t, tId, newSession);
      } catch (error) {
        console.error("Upload error:", error);
        alert("Failed to upload video");
      }
    },
    [meta, sessionId, tasks, pollTask, backend, fps]
  );

  const searchHistory = useCallback(async (): Promise<void> => {
    setHistoryLoading(true);
    try {
      const params = new URLSearchParams();
      if (historySearch.prop) params.append("prop", historySearch.prop);
      if (historySearch.motor) params.append("motor", historySearch.motor);
      if (historySearch.esc) params.append("esc", historySearch.esc);
      const res = await fetch(`${backend}/history/search?${params.toString()}`);
      const data = await res.json();
      setHistoryResults(data.results || []);
    } finally {
      setHistoryLoading(false);
    }
  }, [backend, historySearch]);

  const clearHistorySearch = useCallback((): void => {
    setHistorySearch({ prop: null, motor: null, esc: null });
    fetchAllHistory();
  }, [fetchAllHistory]);

  const openHistoryDetail = useCallback((r: HistoryResult) => {
    setSelectedHistoryResult(r);
    setHistoryDetailOpen(true);
  }, []);

  const closeHistoryDetail = useCallback(() => {
    setSelectedHistoryResult(null);
    setHistoryDetailOpen(false);
  }, []);

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

        {/* ===== ADDED FPS DROPDOWN =====
            Kept inside the same InputRow grid so UI remains unchanged.
            Using native select so it matches MUI TextField behavior.
        */}
        <Grid item xs={12} sm={1}>
          <TextField
            select
            label="FPS"
            value={fps}
            onChange={(e) => setFps(Number(e.target.value))}
            SelectProps={{ native: true }}
            fullWidth
            variant="outlined"
          >
            {[...Array(10)].map((_, i) => (
              <option key={i + 1} value={i + 1}>
                {i + 1}
              </option>
            ))}
          </TextField>
        </Grid>
      </Grid>
    ),
    [meta, handleMetaChange, fps]
  );

  const TabPane = (t: TabKey, title: string): JSX.Element => {
    const st = tasks[t];
    return (
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            {title}
          </Typography>
          <input
            type="file"
            accept="video/*"
            onChange={(e) => handleFilePick(t, e.target.files?.[0] || null)}
          />
          <Button variant="contained" sx={{ ml: 2 }} onClick={() => startUpload(t)}>
            Upload & Start
          </Button>
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

  const HistoryTab = (): JSX.Element => (
    <Box>
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Search Test Results
          </Typography>
          <Grid container spacing={2} sx={{ mb: 2 }}>
            <Grid item xs={12} sm={3}>
              <TextField
                label="Prop Name"
                fullWidth
                value={historySearch.prop || ""}
                onChange={(e) => setHistorySearch((prev) => ({ ...prev, prop: e.target.value || null }))}
                variant="outlined"
              />
            </Grid>
            <Grid item xs={12} sm={3}>
              <TextField
                label="Motor Name"
                fullWidth
                value={historySearch.motor || ""}
                onChange={(e) => setHistorySearch((prev) => ({ ...prev, motor: e.target.value || null }))}
                variant="outlined"
              />
            </Grid>
            <Grid item xs={12} sm={3}>
              <TextField
                label="ESC Name"
                fullWidth
                value={historySearch.esc || ""}
                onChange={(e) => setHistorySearch((prev) => ({ ...prev, esc: e.target.value || null }))}
                variant="outlined"
              />
            </Grid>
            <Grid item xs={12} sm={3}>
              <Box sx={{ display: "flex", gap: 1 }}>
                <Button variant="contained" fullWidth onClick={searchHistory}>
                  Search
                </Button>
                <Button variant="outlined" fullWidth onClick={clearHistorySearch}>
                  Clear
                </Button>
              </Box>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {historyLoading ? (
        <Box sx={{ display: "flex", justifyContent: "center", p: 3 }}>
          <CircularProgress />
        </Box>
      ) : historyResults.length > 0 ? (
        <Paper sx={{ width: "100%", overflow: "auto" }}>
          <Table size="small">
            <TableHead>
              <TableRow sx={{ backgroundColor: "rgba(0, 212, 255, 0.08)" }}>
                <TableCell>Prop</TableCell>
                <TableCell>Motor</TableCell>
                <TableCell>ESC</TableCell>
                <TableCell>Voltage (V)</TableCell>
                <TableCell>Date</TableCell>
                <TableCell>Action</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {historyResults.map((r, i) => (
                <TableRow key={i}>
                  <TableCell>{r.prop_name}</TableCell>
                  <TableCell>{r.motor_name}</TableCell>
                  <TableCell>{r.esc_name}</TableCell>
                  <TableCell>{r.voltage}</TableCell>
                  <TableCell>{new Date(r.created_at).toLocaleString()}</TableCell>
                  <TableCell>
                    <Button variant="outlined" size="small" onClick={() => openHistoryDetail(r)}>
                      View
                    </Button>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </Paper>
      ) : (
        <Typography variant="body2" sx={{ opacity: 0.7, textAlign: "center", p: 3 }}>
          No test results found.
        </Typography>
      )}

      <Dialog open={historyDetailOpen} onClose={closeHistoryDetail} maxWidth="lg" fullWidth>
        <DialogTitle>
          Test Result Details – {selectedHistoryResult?.prop_name || ""}
        </DialogTitle>
        <DialogContent>
          {selectedHistoryResult && (
            <>
              <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>
                Metadata
              </Typography>
              <Grid container spacing={2} sx={{ mb: 2 }}>
                <Grid item xs={6}>
                  <Typography variant="body2">
                    <strong>Motor:</strong> {selectedHistoryResult.motor_name}
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="body2">
                    <strong>ESC:</strong> {selectedHistoryResult.esc_name}
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="body2">
                    <strong>Voltage:</strong> {selectedHistoryResult.voltage}
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="body2">
                    <strong>Date:</strong>{" "}
                    {new Date(selectedHistoryResult.created_at).toLocaleString()}
                  </Typography>
                </Grid>
              </Grid>
              <Typography variant="h6" gutterBottom>
                Test Data
              </Typography>
              {selectedHistoryResult.table_data.length > 0 ? (
                <Paper sx={{ width: "100%", overflow: "auto" }}>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>Time (s)</TableCell>
                        <TableCell>Voltage (V)</TableCell>
                        <TableCell>Current (A)</TableCell>
                        <TableCell>Thrust (G)</TableCell>
                        <TableCell>RPM</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {selectedHistoryResult.table_data.map((row, idx) => (
                        <TableRow key={idx}>
                          <TableCell>{row["Time (s)"]}</TableCell>
                          <TableCell>{row["Voltage (V)"]}</TableCell>
                          <TableCell>{row["Current (A)"]}</TableCell>
                          <TableCell>{row["Thrust (G)"]}</TableCell>
                          <TableCell>{row["RPM"]}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </Paper>
              ) : (
                <Typography>No table data available.</Typography>
              )}
              {selectedHistoryResult.graph_paths.length > 0 && (
                <>
                  <Typography variant="h6" sx={{ mt: 3 }}>
                    Graphs
                  </Typography>
                  {selectedHistoryResult.graph_paths.map((g, idx) => (
                    <Box key={idx} sx={{ mt: 2 }}>
                      <img src={g} alt={`graph-${idx}`} style={{ width: "100%", borderRadius: 8 }} />
                    </Box>
                  ))}
                </>
              )}
              {selectedHistoryResult.csv_path && (
                <Box sx={{ mt: 3 }}>
                  <Button variant="contained" href={selectedHistoryResult.csv_path} download>
                    Download CSV
                  </Button>
                </Box>
              )}
            </>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={closeHistoryDetail}>Close</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );

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
          <Typography variant="h6" sx={{ flexGrow: 1, fontWeight: 700, letterSpacing: 1 }}>
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
        <Tabs
          value={mainTab}
          onChange={handleMainTabChange}
          sx={{ mb: 4, borderBottom: "2px solid rgba(0, 212, 255, 0.1)" }}
        >
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
                <Tabs value={tab} onChange={handleTabChange} sx={{ mt: 2 }}>
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
                    Upload and process your Current/Thrust/RPM videos to see the merged table here.
                  </Typography>
                )}
                {graphs.length > 0 && (
                  <>
                    <Typography variant="h6" sx={{ mt: 3, mb: 2 }}>
                      Graphs
                    </Typography>
                    {graphs.map((g, i) => (
                      <Box key={i} sx={{ mt: 2 }}>
                        <img src={g} alt={`graph-${i}`} style={{ width: "100%", borderRadius: 8 }} />
                      </Box>
                    ))}
                  </>
                )}
              </CardContent>
            </Card>
          </>
        )}

        {mainTab === "history" && <HistoryTab />}
      </Container>
    </ThemeProvider>
  );
}
