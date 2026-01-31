import { Box, Button, Chip, Stack, Typography } from "@mui/material";
import { useState } from "react";

export default function LiveSession() {
  const [isRunning, setIsRunning] = useState(false);

  return (
    <Stack spacing={3}>
      <Typography variant="h4">Live Session</Typography>

      <Box
        sx={{
          height: 360,
          borderRadius: 5,
          bgcolor: "grey.900",
          position: "relative",
          overflow: "hidden",
        }}
      >
        <Box
          sx={{
            position: "absolute",
            inset: 24,
            borderRadius: 4,
            border: "2px dashed",
            borderColor: "grey.700",
          }}
        />
        <Box
          sx={{
            position: "absolute",
            left: 24,
            top: 24,
          }}
        >
          <Chip label="Tracking form" color="primary" size="small" />
        </Box>
        <Box
          sx={{
            position: "absolute",
            inset: 0,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            flexDirection: "column",
            color: "grey.500",
            gap: 0.5,
          }}
        >
          <Typography variant="subtitle1">Camera feed</Typography>
          <Typography variant="caption">Skeleton overlay</Typography>
        </Box>
      </Box>

      <Stack spacing={1}>
        <Typography variant="h6">Feedback</Typography>
        <Typography variant="body1" color="text.secondary">
          Good depth. Keep knees aligned.
        </Typography>
      </Stack>

      <Button
        variant={isRunning ? "outlined" : "contained"}
        onClick={() => setIsRunning((prev) => !prev)}
      >
        {isRunning ? "Stop" : "Start"}
      </Button>
    </Stack>
  );
}
