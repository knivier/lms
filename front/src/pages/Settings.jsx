import { Stack, Typography, Button, Box } from "@mui/material";
import ExitToAppIcon from "@mui/icons-material/ExitToApp";

const isTauri = typeof window !== "undefined" && (window.__TAURI_INTERNALS__ != null || window.__TAURI__ != null);

export default function Settings() {
  async function handleExit() {
    if (!isTauri) return;
    try {
      const { getCurrentWindow } = await import("@tauri-apps/api/window");
      await getCurrentWindow().close();
    } catch (_) {}
  }

  return (
    <Stack spacing={3}>
      <Typography variant="h4">Settings</Typography>
      <Typography variant="body1" color="text.secondary">
        Nothing here yet.
      </Typography>
      {isTauri && (
        <Box sx={{ pt: 2 }}>
          <Button
            variant="contained"
            size="large"
            startIcon={<ExitToAppIcon />}
            onClick={handleExit}
            sx={{
              px: 3,
              py: 1.5,
              borderRadius: 2.5,
              textTransform: "none",
              fontWeight: 600,
              fontSize: "1rem",
              background: "linear-gradient(135deg, #dc2626 0%, #b91c1c 100%)",
              color: "#fff",
              boxShadow: "0 4px 14px rgba(220, 38, 38, 0.4)",
              transition: "all 0.2s ease",
              "&:hover": {
                background: "linear-gradient(135deg, #b91c1c 0%, #991b1b 100%)",
                boxShadow: "0 6px 20px rgba(220, 38, 38, 0.5)",
                transform: "translateY(-2px)",
              },
            }}
          >
            Exit
          </Button>
        </Box>
      )}
    </Stack>
  );
}
