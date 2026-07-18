const API_BASE_URL = typeof window !== 'undefined' 
  ? (window.location.port === '3000'
      ? 'http://localhost:8000/api/v1'  // Route directly to FastAPI during bare-metal run
      : (window.location.origin.includes('localhost') || window.location.origin.includes('127.0.0.1')
          ? 'http://localhost/api/v1'  // Route through NGINX locally in Docker
          : '/api/v1'))
  : 'http://backend:8000/api/v1';

export function getAuthToken(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem("token");
}

export function setAuthToken(token: string) {
  localStorage.setItem("token", token);
}

export function clearAuthToken() {
  localStorage.removeItem("token");
}

// Global fetch wrapper injecting JWT auth header
async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const token = getAuthToken();
  const headers = {
    ...options.headers as any,
  };
  
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...options,
    headers,
  });

  if (response.status === 401) {
    clearAuthToken();
    if (typeof window !== "undefined") {
      window.location.href = "/login";
    }
    throw new Error("Unauthorized");
  }

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ detail: "Request failed" }));
    throw new Error(errorData.detail || "Request failed");
  }

  if (response.status === 204) {
    return {} as T;
  }

  return response.json();
}

export const api = {
  // Auth API
  async login(email: string, password: string): Promise<string> {
    const formData = new URLSearchParams();
    formData.append("username", email);
    formData.append("password", password);

    const data = await request<{ access_token: string }>("/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: formData.toString()
    });
    
    setAuthToken(data.access_token);
    return data.access_token;
  },

  async signup(email: string, password: string): Promise<any> {
    return request("/auth/register", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password })
    });
  },

  async getMe(): Promise<{ email: string; id: string }> {
    return request("/auth/me");
  },

  // Documents API
  async uploadDocument(file: File, modelType: string): Promise<any> {
    const token = getAuthToken();
    const formData = new FormData();
    formData.append("file", file);

    const headers: Record<string, string> = {};
    if (token) headers["Authorization"] = `Bearer ${token}`;

    const response = await fetch(`${API_BASE_URL}/documents/upload?model_type=${modelType}`, {
      method: "POST",
      headers,
      body: formData
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: "Upload failed" }));
      throw new Error(errorData.detail || "Upload failed");
    }
    return response.json();
  },

  async listDocuments(): Promise<any[]> {
    return request("/documents/");
  },

  async getDocument(documentId: string): Promise<any> {
    return request(`/documents/${documentId}`);
  },

  getPageImageUrl(documentId: string, pageNum: number): string {
    const token = getAuthToken();
    return `${API_BASE_URL}/documents/${documentId}/pages/${pageNum}${token ? `?token=${token}` : ''}`;
  },

  // Conversations API
  async listConversations(): Promise<any[]> {
    return request("/chats/conversations");
  },

  async getConversation(conversationId: string): Promise<any> {
    return request(`/chats/conversations/${conversationId}`);
  },

  async createConversation(documentId: string, title?: string): Promise<any> {
    return request("/chats/conversations", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ document_id: documentId, title })
    });
  },

  async deleteConversation(conversationId: string): Promise<any> {
    return request(`/chats/conversations/${conversationId}`, {
      method: "DELETE"
    });
  },

  // Streaming Chat API
  async streamChatMessage(
    conversationId: string,
    message: string,
    modelType: string,
    documentIds: string[] | undefined,
    onCitations: (chunks: any[]) => void,
    onToken: (token: string) => void,
    onDone: () => void,
    onError: (err: string) => void
  ) {
    const token = getAuthToken();
    const headers: Record<string, string> = {
      "Content-Type": "application/json"
    };
    if (token) headers["Authorization"] = `Bearer ${token}`;

    try {
      const response = await fetch(
        `${API_BASE_URL}/chats/conversations/${conversationId}/messages?model_type=${modelType}`,
        {
          method: "POST",
          headers,
          body: JSON.stringify({ content: message, document_ids: documentIds })
        }
      );

      if (!response.ok) {
        const errJson = await response.json().catch(() => ({ detail: "Stream failed" }));
        throw new Error(errJson.detail || "Stream failed");
      }

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      if (!reader) throw new Error("Response body is not readable");

      let buffer = "";
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n\n");
        buffer = lines.pop() || ""; // Keep the last incomplete block in the buffer

        for (const line of lines) {
          if (!line.trim()) continue;

          // Parse event types: citations, token, error, done
          const eventMatch = line.match(/^event:\s*(.+)$/m);
          const dataMatch = line.match(/^data:\s*(.+)$/m);

          if (!eventMatch || !dataMatch) continue;
          
          const eventType = eventMatch[1].trim();
          const dataContent = JSON.parse(dataMatch[1].trim());

          if (eventType === "citations") {
            onCitations(dataContent.chunks);
          } else if (eventType === "token") {
            onToken(dataContent.token);
          } else if (eventType === "error") {
            onError(dataContent.detail);
          } else if (eventType === "done") {
            onDone();
          }
        }
      }
    } catch (err: any) {
      onError(err.message || "Network error");
    }
  },

  // Notes API
  async listNotes(): Promise<any[]> {
    return request("/notes/");
  },

  async createNote(title: string, content: string): Promise<any> {
    return request("/notes/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title, content })
    });
  },

  async updateNote(noteId: string, title: string, content: string): Promise<any> {
    return request(`/notes/${noteId}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title, content })
    });
  },

  async deleteNote(noteId: string): Promise<any> {
    return request(`/notes/${noteId}`, {
      method: "DELETE"
    });
  },

  // Guide Generator API
  async streamGuide(
    documentIds: string[],
    guideType: string,
    modelType: string,
    onToken: (token: string) => void,
    onDone: () => void,
    onError: (err: string) => void
  ) {
    const token = getAuthToken();
    const headers: Record<string, string> = {
      "Content-Type": "application/json"
    };
    if (token) headers["Authorization"] = `Bearer ${token}`;

    try {
      const response = await fetch(
        `${API_BASE_URL}/chats/generate-guide?model_type=${modelType}`,
        {
          method: "POST",
          headers,
          body: JSON.stringify({ document_ids: documentIds, guide_type: guideType })
        }
      );

      if (!response.ok) {
        const errJson = await response.json().catch(() => ({ detail: "Guide generation failed" }));
        throw new Error(errJson.detail || "Guide generation failed");
      }

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      if (!reader) throw new Error("Response body is not readable");

      let buffer = "";
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (!line.trim()) continue;

          const eventMatch = line.match(/^event:\s*(.+)$/m);
          const dataMatch = line.match(/^data:\s*(.+)$/m);

          if (!eventMatch || !dataMatch) continue;

          const eventType = eventMatch[1].trim();
          const dataContent = JSON.parse(dataMatch[1].trim());

          if (eventType === "token") {
            onToken(dataContent.token);
          } else if (eventType === "error") {
            onError(dataContent.detail);
          } else if (eventType === "done") {
            onDone();
          }
        }
      }
    } catch (err: any) {
      onError(err.message || "Network error");
    }
  }
};
