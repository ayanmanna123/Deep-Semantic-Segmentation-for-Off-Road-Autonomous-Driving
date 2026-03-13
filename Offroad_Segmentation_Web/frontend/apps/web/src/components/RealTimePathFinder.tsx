import React, { useRef, useEffect, useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@workspace/ui/components/card";
import { Button } from "@workspace/ui/components/button";
import { Loader2, Camera, CameraOff, Navigation } from "lucide-react";

const WS_URL = "ws://localhost:8001/ws/pathfinder";

export function RealTimePathFinder() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isActive, setIsActive] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const animationFrameRef = useRef<number | null>(null);

  const startCamera = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { facingMode: 'environment', width: 640, height: 480 } 
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setIsActive(true);
        connectWebSocket();
      }
    } catch (err: any) {
      console.error("Camera error:", err);
      setError("Failed to access camera. Please check permissions.");
    } finally {
      setIsLoading(false);
    }
  };

  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }
    if (wsRef.current) {
      wsRef.current.close();
    }
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
    }
    setIsActive(false);
  };

  const connectWebSocket = () => {
    wsRef.current = new WebSocket(WS_URL);
    wsRef.current.onopen = () => {
      console.log("Connected to PathFinder WebSocket");
      processFrame();
    };
    wsRef.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      drawPath(data.path, data.maskWidth, data.maskHeight);
    };
    wsRef.current.onerror = (err) => {
      console.error("WebSocket error:", err);
      setError("WebSocket connection failed.");
    };
  };

  const processFrame = () => {
    if (!isActive || !videoRef.current || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;

    const canvas = document.createElement('canvas');
    canvas.width = 320; // Send smaller frames for speed
    canvas.height = 320;
    const ctx = canvas.getContext('2d');
    if (ctx) {
      ctx.drawImage(videoRef.current, 0, 0, 320, 320);
      const frame = canvas.toDataURL('image/jpeg', 0.7);
      wsRef.current.send(JSON.stringify({ frame }));
    }

    // Process next frame after a slight delay to avoid overwhelming the server
    setTimeout(() => {
        animationFrameRef.current = requestAnimationFrame(processFrame);
    }, 100); 
  };

  const drawPath = (path: [number, number][], maskW: number, maskH: number) => {
    if (!canvasRef.current || !videoRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    canvas.width = videoRef.current.clientWidth;
    canvas.height = videoRef.current.clientHeight;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!path || path.length < 2) return;

    const scaleX = canvas.width / maskW;
    const scaleY = canvas.height / maskH;

    ctx.beginPath();
    ctx.lineWidth = 8;
    ctx.strokeStyle = '#4f46e5'; // Indigo-600
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';

    ctx.moveTo(path[0][0] * scaleX, path[0][1] * scaleY);
    for (let i = 1; i < path.length; i++) {
      ctx.lineTo(path[i][0] * scaleX, path[i][1] * scaleY);
    }
    ctx.stroke();

    // Secondary glow effect
    ctx.lineWidth = 4;
    ctx.strokeStyle = '#818cf8'; // Indigo-400
    ctx.stroke();
  };

  useEffect(() => {
    return () => stopCamera();
  }, []);

  return (
    <Card className="border border-white/5 shadow-2xl bg-slate-900/40 backdrop-blur-xl rounded-2xl overflow-hidden ring-1 ring-white/5">
      <CardHeader className="bg-white/[0.02] border-b border-white/5 flex flex-row items-center justify-between">
        <div>
          <CardTitle className="text-white flex items-center gap-2">
            <Navigation className="h-5 w-5 text-indigo-400" />
            Real-Time Path Finder
          </CardTitle>
          <CardDescription className="text-slate-400">
            Live offroad navigation using camera feed and A* pathfinding.
          </CardDescription>
        </div>
        <Button 
          onClick={isActive ? stopCamera : startCamera}
          variant={isActive ? "destructive" : "default"}
          className={`gap-2 ${!isActive ? 'bg-indigo-600 hover:bg-indigo-700' : ''}`}
          disabled={isLoading}
        >
          {isLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : (isActive ? <CameraOff className="h-4 w-4" /> : <Camera className="h-4 w-4" />)}
          {isActive ? "Stop Navigation" : "Start Live Feed"}
        </Button>
      </CardHeader>
      <CardContent className="p-0 relative bg-black aspect-video flex items-center justify-center">
        {error && (
            <div className="absolute inset-0 z-20 flex items-center justify-center bg-black/80 p-4">
                <p className="text-red-400 font-medium text-center">{error}</p>
            </div>
        )}
        
        <video 
          ref={videoRef} 
          autoPlay 
          playsInline 
          muted 
          className={`w-full h-full object-cover ${isActive ? 'block' : 'hidden'}`}
        />
        <canvas 
          ref={canvasRef} 
          className="absolute inset-0 w-full h-full pointer-events-none z-10"
        />
        
        {!isActive && !isLoading && !error && (
          <div className="flex flex-col items-center gap-4 text-slate-500">
            <div className="p-8 bg-white/[0.02] rounded-full ring-1 ring-white/5">
              <Camera className="h-16 w-16 opacity-10" />
            </div>
            <p className="text-lg font-medium">Camera feed inactive</p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
