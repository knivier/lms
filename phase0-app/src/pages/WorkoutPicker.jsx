import { Card, CardActionArea, CardContent, Stack, Typography } from "@mui/material";
import { useNavigate } from "react-router-dom";

const workouts = [
  { title: "Squat", description: "Lower body strength and control." },
  { title: "Push-up", description: "Upper body endurance and form." },
  { title: "Bicep Curl", description: "Arm strength and tempo." },
];

export default function WorkoutPicker() {
  const navigate = useNavigate();

  return (
    <Stack spacing={3}>
      <Typography variant="h4">Choose Workout</Typography>
      <Stack spacing={2}>
        {workouts.map((workout) => (
          <Card key={workout.title} elevation={3}>
            <CardActionArea onClick={() => navigate("/live")}>
              <CardContent sx={{ py: 3 }}>
                <Typography variant="h6" gutterBottom>
                  {workout.title}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {workout.description}
                </Typography>
              </CardContent>
            </CardActionArea>
          </Card>
        ))}
      </Stack>
    </Stack>
  );
}
