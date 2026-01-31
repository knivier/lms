import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import { CssBaseline, ThemeProvider, createTheme } from "@mui/material";
import App from "./App";
import "./App.css";

const theme = createTheme({
  palette: {
    mode: "light",
    primary: {
      main: "#5B5BD6",
    },
    background: {
      default: "#F7F7FB",
      paper: "#FFFFFF",
    },
  },
  shape: {
    borderRadius: 24,
  },
  typography: {
    fontFamily:
      "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
    h4: { fontWeight: 600 },
    h6: { fontWeight: 600 },
    button: { textTransform: "none", fontWeight: 600 },
  },
  components: {
    MuiButton: {
      defaultProps: { size: "large" },
      styleOverrides: {
        root: { borderRadius: 20, paddingInline: 20, paddingBlock: 12 },
      },
    },
    MuiCard: {
      styleOverrides: { root: { borderRadius: 24 } },
    },
    MuiChip: {
      styleOverrides: { root: { borderRadius: 16 } },
    },
  },
});

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <BrowserRouter>
        <App />
      </BrowserRouter>
    </ThemeProvider>
  </React.StrictMode>,
);
