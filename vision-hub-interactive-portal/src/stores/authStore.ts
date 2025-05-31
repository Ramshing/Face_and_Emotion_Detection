
import { create } from 'zustand';

interface AuthState {
  isLoggedIn: boolean;
  username: string | null;
  login: (username: string, password: string) => boolean;
  logout: () => void;
}

// This is a simple mock auth store. In a real application, 
// you would connect this to a backend authentication service.
export const useAuthStore = create<AuthState>((set) => ({
  isLoggedIn: false,
  username: null,
  login: (username: string, password: string) => {
    // Simple validation, in a real app this would verify against a backend
    if (username && password.length > 3) {
      set({ isLoggedIn: true, username });
      return true;
    }
    return false;
  },
  logout: () => set({ isLoggedIn: false, username: null }),
}));
