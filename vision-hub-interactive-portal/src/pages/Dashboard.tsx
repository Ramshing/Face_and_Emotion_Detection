
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAuthStore } from "@/stores/authStore";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Eye, Camera, LogIn, UserRound } from "lucide-react";

const Dashboard = () => {
  const { username, logout } = useAuthStore();
  const navigate = useNavigate();

  const handleOptionSelect = (option: string) => {
    navigate(`/${option}`);
  };

  return (
    <div className="min-h-screen flex flex-col bg-gradient-to-br from-vision-blue/10 to-vision-purple/10">
      <header className="bg-white/80 backdrop-blur-sm shadow-sm">
        <div className="container mx-auto p-4 flex justify-between items-center">
          <h1 className="text-2xl font-bold text-vision-dark">Vision Hub</h1>
          <div className="flex items-center gap-4">
            <span className="text-sm font-medium">Welcome, {username}</span>
            <Button 
              variant="outline" 
              size="sm" 
              onClick={logout}
              className="flex items-center gap-2"
            >
              <LogIn className="h-4 w-4" />
              Sign Out
            </Button>
          </div>
        </div>
      </header>

      <main className="flex-1 container mx-auto p-4 md:p-8">
        <h2 className="text-3xl font-bold mb-8 animate-fade-in">AI Vision Tools</h2>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* Face Recognition */}
          <Card className="border border-white/20 shadow-lg hover:shadow-xl transition-all duration-300 hover:-translate-y-1 animate-fade-in delay-100">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <UserRound className="h-6 w-6 text-vision-purple" />
                Face and Emotion Recognition
              </CardTitle>
              <CardDescription>
                Identify and verify faces and emotion
              </CardDescription>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-gray-600">
                Accurately detect and identify faces in images or video streams. Perfect for security or automated tagging.
              </p>
            </CardContent>
            <CardFooter>
              <Button 
                className="w-full bg-vision-purple text-white hover:bg-vision-purple/90"
                onClick={() => handleOptionSelect("face-recognition")}
              >
                Start Analysis
              </Button>
            </CardFooter>
          </Card>
        </div>
      </main>
    </div>
  );
};

export default Dashboard;
