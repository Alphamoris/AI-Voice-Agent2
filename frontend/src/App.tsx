
import React, { useState, useCallback, useEffect } from 'react';
import { Box, Container, Paper, Typography, IconButton } from '@mui/material';
import { styled } from '@mui/material/styles';
import MicIcon from '@mui/icons-material/Mic';
import StopIcon from '@mui/icons-material/Stop';
import useWebSocket, { ReadyState } from 'react-use-websocket';
import './App.css';

interface StatusIndicatorProps {
  isActive: boolean;
}

const WaveformContainer = styled(Box)(({ theme }) => ({
  width: '100%',
  height: '120px',
  backgroundColor: theme.palette.grey[100],
  borderRadius: theme.shape.borderRadius,
  position: 'relative',
  overflow: 'hidden',
  marginBottom: theme.spacing(2),
}));

const StatusIndicator = styled('div')<StatusIndicatorProps>(({ theme, isActive }) => ({
  width: '12px',
  height: '12px',
  borderRadius: '50%',
  backgroundColor: isActive ? theme.palette.success.main : theme.palette.grey[400],
  marginRight: theme.spacing(1),
}));

function App() {
  const [isRecording, setIsRecording] = useState<boolean>(false);
  const [mediaRecorder, setMediaRecorder] = useState<MediaRecorder | null>(null);
  const [status, setStatus] = useState<string>('Ready');

  const { sendMessage, lastMessage, readyState } = useWebSocket('ws://localhost:8000/conversation', {
    onOpen: () => setStatus('Connected'),
    onClose: () => setStatus('Disconnected'),
    onError: () => setStatus('Error'),
    shouldReconnect: (closeEvent) => true,
  });

  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      const chunks: BlobPart[] = [];

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          chunks.push(e.data);
          // Send audio chunk to websocket
          const blob = new Blob([e.data], { type: 'audio/webm' });
          sendMessage(blob);
        }
      };

      recorder.onstop = () => {
        stream.getTracks().forEach(track => track.stop());
      };

      recorder.start(100); // Collect 100ms chunks
      setMediaRecorder(recorder);
      setIsRecording(true);
      setStatus('Recording');
    } catch (err) {
      console.error('Error accessing microphone:', err);
      setStatus('Microphone Error');
    }
  }, [sendMessage]);

  const stopRecording = useCallback(() => {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
      mediaRecorder.stop();
      setIsRecording(false);
      setStatus('Ready');
    }
  }, [mediaRecorder]);

  useEffect(() => {
    if (lastMessage && lastMessage.data instanceof Blob) {
      // Handle incoming audio response
      const audio = new Audio(URL.createObjectURL(lastMessage.data));
      audio.play().catch(err => {
        console.error('Error playing audio:', err);
        setStatus('Audio Playback Error');
      });
    }
  }, [lastMessage]);

  const connectionStatus = {
    [ReadyState.CONNECTING]: 'Connecting',
    [ReadyState.OPEN]: 'Connected',
    [ReadyState.CLOSING]: 'Closing',
    [ReadyState.CLOSED]: 'Disconnected',
    [ReadyState.UNINSTANTIATED]: 'Uninstantiated',
  }[readyState];

  return (
    <Container maxWidth="sm" sx={{ py: 4 }}>
      <Paper elevation={3} sx={{ p: 4, borderRadius: 2 }}>
        <Typography variant="h4" gutterBottom align="center">
          AI Voice Agent
        </Typography>
        
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
          <StatusIndicator isActive={readyState === ReadyState.OPEN} />
          <Typography variant="body2" color="text.secondary">
            {status} ({connectionStatus})
          </Typography>
        </Box>

        <WaveformContainer>
          {/* Waveform visualization would go here */}
        </WaveformContainer>

        <Box sx={{ display: 'flex', justifyContent: 'center', gap: 2 }}>
          {!isRecording ? (
            <IconButton
              color="primary"
              size="large"
              onClick={startRecording}
              disabled={readyState !== ReadyState.OPEN}
              sx={{ 
                width: 64, 
                height: 64,
                backgroundColor: 'primary.main',
                color: 'white',
                '&:hover': {
                  backgroundColor: 'primary.dark',
                },
                '&.Mui-disabled': {
                  backgroundColor: 'grey.300',
                  color: 'grey.500',
                }
              }}
            >
              <MicIcon fontSize="large" />
            </IconButton>
          ) : (
            <IconButton
              color="error"
              size="large"
              onClick={stopRecording}
              sx={{ 
                width: 64, 
                height: 64,
                backgroundColor: 'error.main',
                color: 'white',
                '&:hover': {
                  backgroundColor: 'error.dark',
                }
              }}
            >
              <StopIcon fontSize="large" />
            </IconButton>
          )}
        </Box>
      </Paper>
    </Container>
  );
}

export default App;
