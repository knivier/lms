import { useRef, useEffect } from "react";
import {
  AppBar,
  BottomNavigation,
  BottomNavigationAction,
  Box,
  Fade,
  Slide,
  Toolbar,
  Typography,
} from "@mui/material";
import FitnessCenterIcon from "@mui/icons-material/FitnessCenter";
import VideocamOutlinedIcon from "@mui/icons-material/VideocamOutlined";
import SettingsOutlinedIcon from "@mui/icons-material/SettingsOutlined";
import { Routes, Route, useLocation, useNavigate } from "react-router-dom";
import WorkoutPicker from "./pages/WorkoutPicker";
import LiveSession from "./pages/LiveSession";
import Settings from "./pages/Settings";

const titlesByPath = {
  "/": "Workout Picker",
  "/live": "Live Session",
  "/settings": "Settings",
};

const pathOrder = ["/", "/live", "/settings"];

function App() {
  const location = useLocation();
  const navigate = useNavigate();
  const prevPathIndexRef = useRef(pathOrder.indexOf(location.pathname === "/" ? "/" : location.pathname));

  const currentPath =
    location.pathname === "/" ? "/" : location.pathname;
  const pathIndex = pathOrder.indexOf(currentPath);
  const direction = pathIndex >= prevPathIndexRef.current ? "left" : "right";

  useEffect(() => {
    prevPathIndexRef.current = pathIndex;
  }, [pathIndex]);

  return (
    <Box
      sx={{
        minHeight: "100vh",
        display: "flex",
        flexDirection: "column",
        background: "linear-gradient(135deg, #f0f4ff 0%, #e0e7ff 25%, #fce7f3 50%, #fef3c7 75%, #e0f2fe 100%)",
        backgroundAttachment: "fixed",
      }}
    >
      <AppBar position="sticky" color="transparent" elevation={0}>
        <Toolbar sx={{ py: 1.5 }}>
          <Typography variant="h6">
            Kinera
          </Typography>
        </Toolbar>
      </AppBar>

      <Box sx={{ flex: 1, px: 4, py: 4, overflow: "hidden" }}>
        <Slide
          in
          key={location.pathname}
          direction={direction}
          timeout={{ enter: 280, exit: 220 }}
          mountOnEnter
          unmountOnExit
          sx={{ width: "100%" }}
        >
          <Fade in timeout={{ enter: 200, exit: 150 }}>
            <Box sx={{ width: "100%" }}>
              <Routes location={location}>
                <Route path="/" element={<WorkoutPicker />} />
                <Route path="/live" element={<LiveSession />} />
                <Route path="/settings" element={<Settings />} />
              </Routes>
            </Box>
          </Fade>
        </Slide>
      </Box>

      <Box sx={{ px: 2, pb: 2 }}>
        <BottomNavigation
          value={currentPath}
          onChange={(_, value) => navigate(value)}
          showLabels
          sx={{ height: 72, borderRadius: 4, bgcolor: "background.paper" }}
        >
          <BottomNavigationAction
            label="Workouts"
            value="/"
            icon={<FitnessCenterIcon />}
          />
          <BottomNavigationAction
            label="Live"
            value="/live"
            icon={<VideocamOutlinedIcon />}
          />
          <BottomNavigationAction
            label="Settings"
            value="/settings"
            icon={<SettingsOutlinedIcon />}
          />
        </BottomNavigation>
      </Box>
    </Box>
  );
}

export default App;
