import { useState, useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useToast } from "@/components/ui/use-toast";
import { Camera, UserRound, Upload, Download } from "lucide-react";
import io from "socket.io-client";

const socket = io(import.meta.env.VITE_API_URL || "http://localhost:5000");

const FaceRecognition = () => {
  const [isProcessing, setIsProcessing] = useState(false);
  const [isStreamActive, setIsStreamActive] = useState(false);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [processedUrl, setProcessedUrl] = useState<string | null>(null);
  const [streamImage, setStreamImage] = useState<string | null>(null);
  const [emotionData, setEmotionData] = useState<{
    faces_detected?: number;
    emotion?: string;
    confidence?: number;
    blink_count?: number;
    file_type?: string;
  } | null>(null);
  const [videoError, setVideoError] = useState<string | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const navigate = useNavigate();
  const { toast } = useToast();

  // Clean up URLs on unmount or new file upload
  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      if (processedUrl) URL.revokeObjectURL(processedUrl);
    };
  }, [previewUrl, processedUrl]);

  // Handle Socket.IO events
  useEffect(() => {
    socket.on('connect', () => {
      console.log('Connected to backend');
      toast({
        title: "Connected",
        description: "Successfully connected to the backend server.",
      });
    });

    socket.on('connect_error', (error) => {
      console.error('Socket connection error:', error);
      toast({
        variant: "destructive",
        title: "Connection Error",
        description: "Failed to connect to the backend server.",
      });
    });

    socket.on('stream_status', (data) => {
      toast({
        title: data.status === 'error' ? "Error" : "Stream Update",
        description: data.message,
        variant: data.status === 'error' ? "destructive" : "default",
      });
      if (data.status === 'stopped') {
        setIsStreamActive(false);
        setStreamImage(null);
        setEmotionData(null);
      }
    });

    socket.on('processed_frame', (data) => {
      setStreamImage(data.image);
      setEmotionData({
        faces_detected: data.faces_detected,
        emotion: data.emotion,
        confidence: data.confidence,
        blink_count: data.blink_count,
        file_type: 'stream'
      });
    });

    socket.on('analysis_result', (data) => {
      setProcessedUrl(data.image || data.video);
      setEmotionData({
        faces_detected: data.faces_detected,
        emotion: data.emotion,
        confidence: data.confidence,
        blink_count: data.blink_count,
        file_type: data.file_type
      });
      setIsProcessing(false);
      toast({
        title: "Analysis Complete",
        description: `${data.faces_detected} face(s) detected`,
      });
    });

    return () => {
      socket.off('connect');
      socket.off('connect_error');
      socket.off('stream_status');
      socket.off('processed_frame');
      socket.off('analysis_result');
    };
  }, []);

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (!file.type.match('image/*') && !file.type.match('video/*')) {
      toast({
        variant: "destructive",
        title: "Invalid File Type",
        description: "Please upload an image or video file.",
      });
      return;
    }

    if (file.size > 20 * 1024 * 1024) { // 20MB limit
      toast({
        variant: "destructive",
        title: "File Too Large",
        description: "File size must be less than 20MB.",
      });
      return;
    }

    if (previewUrl) URL.revokeObjectURL(previewUrl);
    if (processedUrl) URL.revokeObjectURL(processedUrl);
    setUploadedFile(file);
    const fileUrl = URL.createObjectURL(file);
    setPreviewUrl(fileUrl);
    setProcessedUrl(null);
    setEmotionData(null);
    setVideoError(null);
  };

  const startStream = () => {
    setIsStreamActive(true);
    socket.emit('start_stream');
  };

  const stopStream = () => {
    socket.emit('stop_stream');
  };

  const processFile = async () => {
    if (!uploadedFile) return;

    setIsProcessing(true);
    toast({
      title: "Processing File",
      description: `Analyzing ${uploadedFile.name}...`,
    });

    const formData = new FormData();
    formData.append("file", uploadedFile);

    try {
      const response = await fetch((import.meta.env.VITE_API_URL || 'http://localhost:5000') + '/analyze', {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || "Failed to process file");
      }

      const data = await response.json();
      setProcessedUrl(data.image || data.video);
      setEmotionData({
        faces_detected: data.faces_detected,
        emotion: data.emotion,
        confidence: data.confidence,
        blink_count: data.blink_count,
        file_type: data.file_type
      });
      setIsProcessing(false);
      toast({
        title: "Analysis Complete",
        description: `${data.faces_detected} face(s) detected`,
      });
    } catch (err: any) {
      setIsProcessing(false);
      setVideoError(err.message);
      toast({
        variant: "destructive",
        title: "Processing Failed",
        description: err.message,
      });
    }
  };

  const downloadVideo = () => {
    if (!processedUrl || emotionData?.file_type !== 'video') return;
    const link = document.createElement('a');
    link.href = processedUrl;
    link.download = 'processed_video.mp4';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="min-h-screen flex flex-col bg-gradient-to-br from-vision-blue/10 to-vision-purple/10">
      <header className="bg-white/80 backdrop-blur-sm shadow-sm">
        <div className="container mx-auto p-4 flex justify-between items-center">
          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => navigate("/dashboard")}
              className="flex items-center gap-2"
            >
              ← Back to Dashboard
            </Button>
          </div>
          <h1 className="text-xl font-bold text-vision-dark flex items-center">
            <UserRound className="mr-2 h-5 w-5 text-vision-purple" />
            Face Recognition
          </h1>
          <div className="w-[100px]"></div>
        </div>
      </header>

      <main className="flex-1 container mx-auto p-4 md:p-8">
        <Tabs defaultValue="upload" className="w-full animate-fade-in">
          <TabsList className="grid w-full max-w-md mx-auto grid-cols-2">
            <TabsTrigger value="upload" className="flex items-center gap-2">
              <Upload className="h-4 w-4" />
              Upload File
            </TabsTrigger>
            <TabsTrigger value="stream" className="flex items-center gap-2">
              <Camera className="h-4 w-4" />
              Live Stream
            </TabsTrigger>
          </TabsList>

          <TabsContent value="upload" className="mt-6">
            <Card className="max-w-4xl mx-auto border border-white/20 shadow-lg">
              <CardContent className="p-6">
                <div className="space-y-4">
                  <div className="flex flex-col items-center justify-center border-2 border-dashed border-gray-300 rounded-lg p-8 bg-gray-50">
                    <input
                      type="file"
                      id="file-upload-face"
                      className="hidden"
                      accept="image/*,video/*"
                      onChange={handleFileUpload}
                    />
                    <label
                      htmlFor="file-upload-face"
                      className="cursor-pointer flex flex-col items-center justify-center"
                    >
                      <Upload className="h-10 w-10 text-gray-400 mb-2" />
                      <p className="text-sm font-medium">Click to upload an image or video</p>
                      <p className="text-xs text-gray-500 mt-1">SVG, PNG, JPG, or MP4 (max. 20MB)</p>
                    </label>
                  </div>

                  {previewUrl && (
                    <div className="mt-6">
                      <h3 className="text-lg font-medium mb-2">Original</h3>
                      <div className="rounded-lg overflow-hidden border border-gray-200 bg-black aspect-video flex items-center justify-center">
                        {uploadedFile?.type.match('image/*') ? (
                          <img
                            src={previewUrl}
                            alt="Preview"
                            className="max-h-full object-contain"
                          />
                        ) : (
                          <video
                            src={previewUrl}
                            controls
                            className="w-full h-full object-contain"
                          />
                        )}
                      </div>
                    </div>
                  )}

                  {processedUrl && (
                    <div className="mt-6">
                      <h3 className="text-lg font-medium mb-2">Processed Result</h3>
                      <div className="rounded-lg overflow-hidden border border-gray-200 bg-black aspect-video flex items-center justify-center relative">
                        {emotionData?.file_type === 'image' ? (
                          <img
                            src={processedUrl}
                            alt="Processed"
                            className="max-h-full object-contain"
                          />
                        ) : (
                          <video
                            ref={videoRef}
                            src={processedUrl}
                            controls
                            autoPlay
                            muted
                            className="w-full h-full object-contain"
                            onError={(e) => {
                              const error = e.currentTarget.error;
                              const errorMessage = error?.message || "Unknown error";
                              console.error('Video playback error:', errorMessage, error);
                              setVideoError(`Video error: ${errorMessage}`);
                              toast({
                                variant: "destructive",
                                title: "Video Playback Error",
                                description: "Could not play the processed video. Try downloading it.",
                              });
                            }}
                            onLoadedData={() => {
                              setVideoError(null);
                              toast({
                                title: "Video Loaded",
                                description: "Processed video loaded successfully.",
                              });
                            }}
                          />
                        )}
                        {videoError && (
                          <div className="absolute inset-0 flex items-center justify-center bg-black/70 text-white p-4 text-center">
                            <div>
                              <p className="font-bold mb-2">Error playing video</p>
                              <p className="text-sm mb-4">{videoError}</p>
                              {emotionData?.file_type === 'video' && (
                                <Button
                                  variant="secondary"
                                  size="sm"
                                  onClick={downloadVideo}
                                  className="flex items-center gap-2"
                                >
                                  <Download className="h-4 w-4" />
                                  Download Video
                                </Button>
                              )}
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  )}

                  {emotionData && (
                    <div className="mt-4 p-4 bg-gray-50 rounded-lg">
                      <h4 className="font-medium">Analysis Results:</h4>
                      <ul className="mt-2 space-y-1">
                        <li>Faces detected: {emotionData.faces_detected ?? "N/A"}</li>
                        <li>Emotion: {emotionData.emotion ?? "N/A"}</li>
                        <li>Confidence: {emotionData.confidence ? `${(emotionData.confidence * 100).toFixed(2)}%` : "N/A"}</li>
                        <li>Blink count: {emotionData.blink_count ?? "N/A"}</li>
                        <li>File type: {emotionData.file_type ?? "N/A"}</li>
                      </ul>
                    </div>
                  )}

                  <div className="mt-4 flex justify-end">
                    <Button
                      className="bg-vision-purple text-white"
                      onClick={processFile}
                      disabled={isProcessing || !uploadedFile}
                    >
                      {isProcessing ? "Processing..." : "Start Analysis"}
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="stream" className="mt-6">
            <Card className="max-w-4xl mx-auto border border-white/20 shadow-lg">
              <CardContent className="p-6">
                <div className="space-y-4">
                  <div className="rounded-lg overflow-hidden border border-gray-200 bg-black aspect-video flex items-center justify-center">
                    {isStreamActive ? (
                      <div className="relative w-full h-full">
                        {streamImage ? (
                          <img
                            src={streamImage}
                            alt="Live stream"
                            className="w-full h-full object-contain"
                          />
                        ) : (
                          <div className="absolute inset-0 flex items-center justify-center">
                            <div className="text-white text-xl">
                              Starting camera stream...
                            </div>
                          </div>
                        )}
                        {emotionData?.file_type === 'stream' && (
                          <div className="absolute top-4 right-4 bg-black/70 text-white px-2 py-1 rounded text-sm font-mono">
                            Faces: {emotionData.faces_detected ?? "N/A"} |
                            Emotion: {emotionData.emotion ?? "N/A"} |
                            Confidence: {emotionData.confidence ? `${(emotionData.confidence * 100).toFixed(2)}%` : "N/A"} |
                            Blinks: {emotionData.blink_count ?? "N/A"}
                          </div>
                        )}
                      </div>
                    ) : (
                      <div className="flex flex-col items-center justify-center text-white">
                        <Camera className="h-10 w-10 mb-2" />
                        <p>Camera stream not active</p>
                      </div>
                    )}
                  </div>

                  <div className="mt-4 flex justify-end">
                    {isStreamActive ? (
                      <Button
                        className="bg-red-500 text-white hover:bg-red-600"
                        onClick={stopStream}
                      >
                        Stop Stream
                      </Button>
                    ) : (
                      <Button
                        className="bg-vision-purple text-white"
                        onClick={startStream}
                        disabled={isProcessing}
                      >
                        Start Live Stream
                      </Button>
                    )}
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </main>
    </div>
  );
};

export default FaceRecognition;