"use client";

import { useEffect, useState, useRef } from "react";
import { useRouter } from "next/navigation";
import { api, getAuthToken, clearAuthToken } from "@/lib/api";
import { 
  FileText, MessageSquare, LogOut, Upload, Send, 
  ChevronLeft, ChevronRight, Settings, Loader2, Sparkles, Trash2, ArrowRight,
  Pin, Plus, Play, Square, Headphones, HelpCircle, Check, X, BookOpen, FileCheck
} from "lucide-react";

// Helper function to render simple markdown formatting
function formatMarkdown(text: string) {
  if (!text) return null;
  
  const lines = text.split("\n");
  return lines.map((line, idx) => {
    let cleanLine = line.trim();
    if (!cleanLine) return <div key={idx} className="h-2" />;
    
    // Check for Headings
    if (cleanLine.startsWith("### ")) {
      return <h4 key={idx} className="text-sm font-bold text-white mt-4 mb-2">{cleanLine.slice(4)}</h4>;
    }
    if (cleanLine.startsWith("## ")) {
      return <h3 key={idx} className="text-base font-bold text-white mt-5 mb-2">{cleanLine.slice(3)}</h3>;
    }
    if (cleanLine.startsWith("# ")) {
      return <h2 key={idx} className="text-lg font-bold text-white mt-6 mb-3">{cleanLine.slice(2)}</h2>;
    }
    
    // Check for list items
    const isBullet = cleanLine.startsWith("- ") || cleanLine.startsWith("* ");
    if (isBullet) {
      cleanLine = cleanLine.slice(2);
    }
    
    // Parse bold tags **text**
    const parts = cleanLine.split("**");
    const content = parts.map((part, pIdx) => {
      if (pIdx % 2 === 1) {
        return <strong key={pIdx} className="font-bold text-white">{part}</strong>;
      }
      return part;
    });

    if (isBullet) {
      return (
        <li key={idx} className="list-disc ml-5 text-xs text-gray-300 leading-relaxed my-1">
          {content}
        </li>
      );
    }

    return (
      <p key={idx} className="text-xs text-gray-300 leading-relaxed my-1.5">
        {content}
      </p>
    );
  });
}

interface ScriptLine {
  speaker: "Dan" | "Lisa";
  text: string;
}

function parsePodcastScript(text: string): ScriptLine[] {
  const lines = text.split("\n");
  const script: ScriptLine[] = [];
  for (let line of lines) {
    line = line.trim();
    if (!line) continue;
    
    if (line.match(/^\*\*Dan:\*\*/i) || line.startsWith("Dan:")) {
      const cleaned = line.replace(/^\*\*Dan:\*\*|^\*\*Dan:\s*\*\*|^Dan:/i, "").trim();
      script.push({ speaker: "Dan", text: cleaned });
    } else if (line.match(/^\*\*Lisa:\*\*/i) || line.startsWith("Lisa:")) {
      const cleaned = line.replace(/^\*\*Lisa:\*\*|^\*\*Lisa:\s*\*\*|^Lisa:/i, "").trim();
      script.push({ speaker: "Lisa", text: cleaned });
    } else if (script.length > 0) {
      // Append text to previous speaker turn
      script[script.length - 1].text += " " + line;
    }
  }
  return script;
}

export default function WorkspacePage() {
  const router = useRouter();
  const messagesEndRef = useRef<HTMLDivElement>(null);
  
  // Auth State
  const [user, setUser] = useState<any>(null);
  const [authChecked, setAuthChecked] = useState(false);

  // Conversations & Docs State
  const [conversations, setConversations] = useState<any[]>([]);
  const [activeConv, setActiveConv] = useState<any | null>(null);
  const [messages, setMessages] = useState<any[]>([]);
  const [documents, setDocuments] = useState<any[]>([]);
  
  // Notes State
  const [notes, setNotes] = useState<any[]>([]);
  const [showNoteEditor, setShowNoteEditor] = useState(false);
  const [noteForm, setNoteForm] = useState({ id: "", title: "", content: "" });

  // Source selection state (checked documents for RAG query)
  const [selectedDocIds, setSelectedDocIds] = useState<string[]>([]);
  
  // Center column view controller: active doc shows PDF, null shows Notes board / Guide hub
  const [activeDoc, setActiveDoc] = useState<any | null>(null);
  const [currentPage, setCurrentPage] = useState<number>(1);
  const [activeTab, setActiveTab] = useState<"notes" | "guide">("notes");

  // Input & Settings State
  const [inputText, setInputText] = useState("");
  const [modelType, setModelType] = useState("OpenAI");
  const [uploading, setUploading] = useState(false);
  const [streaming, setStreaming] = useState(false);
  
  // Temp loading / feedback states
  const [uploadError, setUploadError] = useState("");
  const [currentStreamingText, setCurrentStreamingText] = useState("");
  const [activeCitations, setActiveCitations] = useState<any[]>([]);

  // Guide Generator State
  const [guideText, setGuideText] = useState("");
  const [guideStreaming, setGuideStreaming] = useState(false);
  const [activeGuideType, setActiveGuideType] = useState<string | null>(null);

  // Audio Podcast State
  const [isPlayingPodcast, setIsPlayingPodcast] = useState(false);
  const [currentPodcastLine, setCurrentPodcastLine] = useState(-1);
  const utteranceRef = useRef<any>(null);

  // 1. Authenticate user on mount
  useEffect(() => {
    const checkAuth = async () => {
      const token = getAuthToken();
      if (!token) {
        router.push("/login");
        return;
      }
      try {
        const userData = await api.getMe();
        setUser(userData);
        setAuthChecked(true);
      } catch (err) {
        clearAuthToken();
        router.push("/login");
      }
    };
    checkAuth();
  }, [router]);

  // 2. Fetch initial data
  useEffect(() => {
    if (!authChecked) return;

    const loadData = async () => {
      try {
        const [convList, docList, noteList] = await Promise.all([
          api.listConversations(),
          api.listDocuments(),
          api.listNotes()
        ]);
        setConversations(convList);
        setDocuments(docList);
        setNotes(noteList);
        
        // Auto-select all ready documents by default
        if (docList.length > 0) {
          const readyDocs = docList.filter(d => d.status === "ready").map(d => d.id);
          setSelectedDocIds(readyDocs);
        }

        // Auto-select first conversation if exists
        if (convList.length > 0) {
          selectConversation(convList[0].id, docList);
        }
      } catch (err) {
        console.error("Failed to load initial data", err);
      }
    };
    loadData();
  }, [authChecked]);

  // Scroll to chat bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, currentStreamingText]);

  // Reload document lists (polling for document ready status)
  useEffect(() => {
    if (!authChecked) return;
    
    const hasProcessing = documents.some(d => d.status === "processing");
    if (!hasProcessing) return;

    const interval = setInterval(async () => {
      const docList = await api.listDocuments();
      setDocuments(docList);
      
      // Update selected document object if its status changes
      if (activeDoc) {
        const updated = docList.find(d => d.id === activeDoc.id);
        if (updated) setActiveDoc(updated);
      }
    }, 4000);

    return () => clearInterval(interval);
  }, [documents, activeDoc, authChecked]);

  // Stop TTS speech on component unmount
  useEffect(() => {
    return () => {
      if (typeof window !== "undefined") {
        window.speechSynthesis.cancel();
      }
    };
  }, []);

  // Actions
  const handleLogout = () => {
    if (typeof window !== "undefined") {
      window.speechSynthesis.cancel();
    }
    clearAuthToken();
    router.push("/login");
  };

  const selectConversation = async (id: string, currentDocs?: any[]) => {
    setStreaming(false);
    setCurrentStreamingText("");
    setActiveCitations([]);
    try {
      const detail = await api.getConversation(id);
      setActiveConv(detail);
      setMessages(detail.messages || []);
      
      const docs = currentDocs || documents;
      // Auto-set the active document linked to the chat
      if (detail.document_id) {
        const doc = docs.find(d => d.id === detail.document_id);
        if (doc) {
          setActiveDoc(doc);
          setCurrentPage(1);
          // Ensure it is checked
          setSelectedDocIds(prev => prev.includes(doc.id) ? prev : [...prev, doc.id]);
        }
      }
    } catch (err) {
      console.error("Failed to load conversation messages", err);
    }
  };

  const handleCreateChat = async (docId?: string, filename?: string) => {
    try {
      const title = filename ? `Chat: ${filename.slice(0, 20)}` : "Workspace Notebook Chat";
      const newConv = await api.createConversation(docId || "", title);
      setConversations(prev => [newConv, ...prev]);
      setActiveConv(newConv);
      setMessages([]);
      
      if (docId) {
        const doc = documents.find(d => d.id === docId);
        setActiveDoc(doc || null);
        setCurrentPage(1);
      }
    } catch (err) {
      console.error("Failed to build chat session", err);
    }
  };

  const handleDeleteChat = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    try {
      await api.deleteConversation(id);
      setConversations(prev => prev.filter(c => c.id !== id));
      if (activeConv?.id === id) {
        setActiveConv(null);
        setMessages([]);
        setActiveDoc(null);
      }
    } catch (err) {
      console.error("Failed to delete chat thread", err);
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploading(true);
    setUploadError("");

    try {
      const doc = await api.uploadDocument(file, modelType);
      setDocuments(prev => [doc, ...prev]);
      
      // Auto-select the document checkbox
      setSelectedDocIds(prev => [...prev, doc.id]);
      
      // Auto build a conversation thread for the uploaded document
      await handleCreateChat(doc.id, doc.filename);
    } catch (err: any) {
      setUploadError(err.message || "Failed to process file upload.");
    } finally {
      setUploading(false);
    }
  };

  const handleToggleDocSelect = (docId: string, status: string) => {
    if (status !== "ready") return;
    setSelectedDocIds(prev =>
      prev.includes(docId) ? prev.filter(id => id !== docId) : [...prev, docId]
    );
  };

  const handleSelectAllDocs = () => {
    const readyDocIds = documents.filter(d => d.status === "ready").map(d => d.id);
    setSelectedDocIds(readyDocIds);
  };

  const handleDeselectAllDocs = () => {
    setSelectedDocIds([]);
  };

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputText.trim() || streaming) return;

    // Check if we have an active conversation, if not create one
    let currentConv = activeConv;
    if (!currentConv) {
      try {
        const title = `Chat: ${inputText.slice(0, 20)}...`;
        currentConv = await api.createConversation(selectedDocIds[0] || "", title);
        setConversations(prev => [currentConv, ...prev]);
        setActiveConv(currentConv);
      } catch (err) {
        console.error("Failed to create conversation for message", err);
        return;
      }
    }

    const userText = inputText;
    setInputText("");
    
    // Render user message instantly
    const userMsgPlaceholder = {
      id: "temp-user-id-" + Date.now(),
      role: "user",
      content: userText,
      created_at: new Date().toISOString()
    };
    setMessages(prev => [...prev, userMsgPlaceholder]);
    
    setStreaming(true);
    setCurrentStreamingText("");
    setActiveCitations([]);

    try {
      await api.streamChatMessage(
        currentConv.id,
        userText,
        modelType,
        selectedDocIds,
        (chunks) => {
          setActiveCitations(chunks);
          if (chunks.length > 0) {
            // Find document and display page in PDF viewer
            const chunkDoc = documents.find(d => d.id === chunks[0].document_id || d.filename === chunks[0].filename);
            if (chunkDoc) {
              setActiveDoc(chunkDoc);
              setCurrentPage(chunks[0].page);
            }
          }
        },
        (token) => {
          setCurrentStreamingText(prev => prev + token);
        },
        async () => {
          setStreaming(false);
          const detail = await api.getConversation(currentConv.id);
          setMessages(detail.messages || []);
          setCurrentStreamingText("");
          setActiveCitations([]);
          
          // Refresh conversation list titles
          const convList = await api.listConversations();
          setConversations(convList);
        },
        (err) => {
          setStreaming(false);
          setErrorAlert(err);
        }
      );
    } catch (err: any) {
      setStreaming(false);
      setErrorAlert(err.message || "Error communicating with server.");
    }
  };

  const setErrorAlert = (msg: string) => {
    const errorMsgPlaceholder = {
      id: "temp-err-id-" + Date.now(),
      role: "assistant",
      content: `❌ Stream Error: ${msg}`,
      created_at: new Date().toISOString()
    };
    setMessages(prev => [...prev, errorMsgPlaceholder]);
  };

  // Nav helpers for PDF
  const handlePrevPage = () => {
    if (currentPage > 1) setCurrentPage(p => p - 1);
  };

  const handleNextPage = () => {
    if (activeDoc && currentPage < activeDoc.page_count) {
      setCurrentPage(p => p + 1);
    }
  };

  // Note CRUD Actions
  const handleOpenNoteEditor = (note?: any) => {
    if (note) {
      setNoteForm({ id: note.id, title: note.title, content: note.content });
    } else {
      setNoteForm({ id: "", title: "", content: "" });
    }
    setShowNoteEditor(true);
  };

  const handleSaveNote = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!noteForm.content.trim()) return;
    try {
      if (noteForm.id) {
        const updated = await api.updateNote(noteForm.id, noteForm.title || "Untitled Note", noteForm.content);
        setNotes(prev => prev.map(n => n.id === updated.id ? updated : n));
      } else {
        const created = await api.createNote(noteForm.title || "Untitled Note", noteForm.content);
        setNotes(prev => [created, ...prev]);
      }
      setShowNoteEditor(false);
    } catch (err) {
      console.error("Failed to save note", err);
    }
  };

  const handleDeleteNote = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    try {
      await api.deleteNote(id);
      setNotes(prev => prev.filter(n => n.id !== id));
    } catch (err) {
      console.error("Failed to delete note", err);
    }
  };

  const handlePinResponse = async (content: string) => {
    try {
      const title = `Pinned Response (${new Date().toLocaleDateString()})`;
      const note = await api.createNote(title, content);
      setNotes(prev => [note, ...prev]);
      setActiveTab("notes");
      setActiveDoc(null); // Switch to notes tab workspace view
    } catch (err) {
      console.error("Failed to pin response to notes", err);
    }
  };

  // Guide Generator Actions
  const handleGenerateGuide = async (type: string) => {
    if (selectedDocIds.length === 0) return;
    
    // Stop playing podcast TTS
    if (isPlayingPodcast) {
      window.speechSynthesis.cancel();
      setIsPlayingPodcast(false);
      setCurrentPodcastLine(-1);
    }

    setActiveGuideType(type);
    setGuideText("");
    setGuideStreaming(true);
    setActiveTab("guide");
    setActiveDoc(null); // Return to Workspace Hub to see guide stream

    try {
      await api.streamGuide(
        selectedDocIds,
        type,
        modelType,
        (token) => {
          setGuideText(prev => prev + token);
        },
        () => {
          setGuideStreaming(false);
        },
        (err) => {
          setGuideStreaming(false);
          setGuideText(prev => prev + `\n\n❌ Guide Error: ${err}`);
        }
      );
    } catch (err: any) {
      setGuideStreaming(false);
      setGuideText(`Error starting guide generation: ${err.message || err}`);
    }
  };

  const handleSaveGuideAsNote = async () => {
    if (!guideText) return;
    try {
      const typeLabel = activeGuideType ? activeGuideType.charAt(0).toUpperCase() + activeGuideType.slice(1).replace("_", " ") : "Guide";
      const title = `Saved ${typeLabel} - ${new Date().toLocaleDateString()}`;
      const note = await api.createNote(title, guideText);
      setNotes(prev => [note, ...prev]);
      setActiveTab("notes");
    } catch (err) {
      console.error("Failed to save guide as note", err);
    }
  };

  // Text-To-Speech Podcast Audio Player
  const podcastScript = activeGuideType === "podcast" ? parsePodcastScript(guideText) : [];

  const handlePlayPodcast = () => {
    if (isPlayingPodcast) {
      window.speechSynthesis.cancel();
      setIsPlayingPodcast(false);
      setCurrentPodcastLine(-1);
      return;
    }
    
    if (podcastScript.length === 0) return;
    setIsPlayingPodcast(true);
    playLine(0);
  };

  const playLine = (index: number) => {
    if (index >= podcastScript.length) {
      setIsPlayingPodcast(false);
      setCurrentPodcastLine(-1);
      return;
    }
    
    setCurrentPodcastLine(index);
    const line = podcastScript[index];
    
    const utterance = new SpeechSynthesisUtterance(line.text);
    utteranceRef.current = utterance;
    
    // Choose voice parameters
    const voices = window.speechSynthesis.getVoices();
    if (line.speaker === "Dan") {
      const maleVoice = voices.find(v => 
        v.name.toLowerCase().includes("male") || 
        v.name.toLowerCase().includes("david") || 
        v.name.toLowerCase().includes("google us english")
      );
      if (maleVoice) utterance.voice = maleVoice;
      utterance.pitch = 0.9;
      utterance.rate = 1.0;
    } else {
      const femaleVoice = voices.find(v => 
        v.name.toLowerCase().includes("female") || 
        v.name.toLowerCase().includes("zira") || 
        v.name.toLowerCase().includes("google uk english female")
      );
      if (femaleVoice) utterance.voice = femaleVoice;
      utterance.pitch = 1.15;
      utterance.rate = 1.05;
    }
    
    utterance.onend = () => {
      playLine(index + 1);
    };
    
    utterance.onerror = () => {
      setIsPlayingPodcast(false);
      setCurrentPodcastLine(-1);
    };
    
    window.speechSynthesis.speak(utterance);
  };

  const handleOpenDocFromCitation = (docId: string, pageNum: number) => {
    const doc = documents.find(d => d.id === docId);
    if (doc) {
      setActiveDoc(doc);
      setCurrentPage(pageNum);
    }
  };

  if (!authChecked) {
    return (
      <div className="flex h-screen items-center justify-center bg-[#070b13]">
        <Loader2 className="h-8 w-8 animate-spin text-indigo-400" />
      </div>
    );
  }

  return (
    <div className="flex h-screen bg-[#070b13] text-[#e2e8f0] overflow-hidden font-sans">
      
      {/* 1. LEFT SIDEBAR PANEL (Sources & Document Picker) */}
      <aside className="w-80 bg-[#0c1222] border-r border-white/5 flex flex-col justify-between select-none shrink-0">
        <div className="flex flex-col flex-1 overflow-y-auto min-h-0">
          
          {/* Header */}
          <div className="p-5 border-b border-white/5 flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Sparkles className="h-5 w-5 text-indigo-400" />
              <span className="font-extrabold tracking-tight bg-gradient-to-r from-sky-400 via-indigo-400 to-purple-400 bg-clip-text text-transparent">
                NotebookLM RAG
              </span>
            </div>
            <span className="text-[10px] uppercase font-bold tracking-widest bg-indigo-500/10 border border-indigo-500/20 px-2 py-0.5 rounded text-indigo-300">
              v1.5
            </span>
          </div>

          {/* Upload Button */}
          <div className="p-4">
            <label className="flex items-center justify-center border border-dashed border-white/10 hover:border-indigo-500/50 bg-[#0f182c] hover:bg-[#131f3b] py-3.5 rounded-xl cursor-pointer transition-all">
              <Upload className="h-4 w-4 text-indigo-400 mr-2" />
              <span className="text-xs font-semibold text-gray-300">
                {uploading ? "Processing PDF..." : "Add Source PDF"}
              </span>
              <input 
                type="file" 
                accept=".pdf" 
                className="hidden" 
                onChange={handleFileUpload} 
                disabled={uploading} 
              />
            </label>
            {uploadError && (
              <p className="text-[10px] text-red-400 mt-2 text-center">{uploadError}</p>
            )}
          </div>

          {/* Source Select Header */}
          <div className="px-4 py-1.5 flex items-center justify-between">
            <p className="text-[10px] font-bold text-gray-400 uppercase tracking-wider">Sources ({selectedDocIds.length} active)</p>
            <div className="flex items-center space-x-2 text-[9px] font-bold text-indigo-400 uppercase">
              <button onClick={handleSelectAllDocs} className="hover:underline">All</button>
              <span className="text-gray-600">|</span>
              <button onClick={handleDeselectAllDocs} className="hover:underline">None</button>
            </div>
          </div>

          {/* Sources List */}
          <div className="px-4 pb-4 space-y-1 overflow-y-auto max-h-52 border-b border-white/5">
            {documents.length === 0 ? (
              <p className="text-[11px] text-gray-500 italic py-2 text-center">No source PDFs uploaded</p>
            ) : (
              documents.map((doc) => (
                <div
                  key={doc.id}
                  className={`flex items-center space-x-2 p-2 rounded-lg transition-all ${
                    activeDoc?.id === doc.id
                      ? "bg-indigo-500/10 text-indigo-300 border border-indigo-500/20"
                      : "hover:bg-white/5 text-gray-300"
                  } ${doc.status === "processing" ? "opacity-60" : ""}`}
                >
                  {/* Source selection Checkbox */}
                  <input
                    type="checkbox"
                    checked={selectedDocIds.includes(doc.id)}
                    disabled={doc.status !== "ready"}
                    onChange={() => handleToggleDocSelect(doc.id, doc.status)}
                    className="w-3.5 h-3.5 accent-indigo-500 rounded border-gray-600 cursor-pointer disabled:cursor-not-allowed"
                  />
                  
                  {/* Clickable filename to load PDF in central view */}
                  <div
                    onClick={() => doc.status === "ready" && setActiveDoc(doc)}
                    className="flex items-center space-x-2 truncate flex-1 cursor-pointer"
                  >
                    <FileText className={`h-3.5 w-3.5 shrink-0 ${selectedDocIds.includes(doc.id) ? "text-indigo-400" : "text-gray-500"}`} />
                    <span className="text-[11px] truncate font-medium">{doc.filename}</span>
                  </div>

                  {doc.status === "processing" ? (
                    <Loader2 className="h-3 w-3 animate-spin text-indigo-400" />
                  ) : (
                    <span className="text-[9px] text-gray-500 bg-black/30 px-1.5 py-0.5 rounded font-mono">{doc.page_count}p</span>
                  )}
                </div>
              ))
            )}
          </div>

          {/* Conversation List */}
          <div className="px-4 py-3 flex-1 flex flex-col min-h-0">
            <p className="text-[10px] font-bold text-gray-400 uppercase tracking-wider mb-2">Notebook Chats</p>
            <div className="space-y-1 flex-1 overflow-y-auto">
              <div
                onClick={() => handleCreateChat()}
                className="flex items-center space-x-2 p-2 rounded-lg cursor-pointer text-gray-400 hover:text-white border border-dashed border-white/5 hover:border-white/20 transition-all text-xs"
              >
                <Plus className="h-3.5 w-3.5" />
                <span className="font-semibold">New Chat Thread</span>
              </div>
              {conversations.length === 0 ? (
                <p className="text-[11px] text-gray-600 italic py-4 text-center">No active threads</p>
              ) : (
                conversations.map((conv) => (
                  <div
                    key={conv.id}
                    onClick={() => selectConversation(conv.id)}
                    className={`flex items-center justify-between p-2.5 rounded-xl cursor-pointer group transition-all ${
                      activeConv?.id === conv.id
                        ? "bg-[#16213e] text-white border border-indigo-500/20"
                        : "hover:bg-white/5 text-gray-400 hover:text-gray-300"
                    }`}
                  >
                    <div className="flex items-center space-x-2 truncate">
                      <MessageSquare className="h-3.5 w-3.5 shrink-0 text-indigo-400" />
                      <span className="text-xs truncate font-medium">{conv.title}</span>
                    </div>
                    <button
                      onClick={(e) => handleDeleteChat(conv.id, e)}
                      className="opacity-0 group-hover:opacity-100 hover:text-red-400 p-0.5 rounded transition-opacity"
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                    </button>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>

        {/* User profile footer */}
        <div className="p-4 border-t border-white/5 flex items-center justify-between bg-[#080d1a]">
          <div className="truncate flex-1 mr-2">
            <p className="text-[9px] text-gray-500">Workspace Tenant</p>
            <p className="text-xs font-semibold text-white truncate">{user?.email}</p>
          </div>
          <button 
            onClick={handleLogout}
            className="p-2 hover:bg-red-500/10 hover:text-red-400 rounded-xl transition-colors text-gray-400"
          >
            <LogOut className="h-4 w-4" />
          </button>
        </div>
      </aside>

      {/* 2. CENTRAL WORKSPACE HUB (Toggles: PDF view OR Notebook guides & notes grid) */}
      <main className="flex-1 flex min-w-0 bg-[#090f1e] overflow-hidden">
        
        {activeDoc ? (
          
          /* 2.1 ACTIVE SOURCE PDF VIEWER */
          <section className="flex-1 flex flex-col border-r border-white/5 bg-black/10 relative">
            
            {/* PDF Viewer Header */}
            <div className="p-4 border-b border-white/5 flex items-center justify-between bg-[#0b1224]">
              <div className="flex items-center space-x-2 truncate">
                <FileText className="h-4 w-4 text-indigo-400" />
                <h2 className="text-xs font-bold truncate text-white">{activeDoc.filename}</h2>
              </div>
              
              {/* Pagination controls */}
              <div className="flex items-center space-x-3 bg-gray-900/60 border border-white/5 px-2.5 py-1 rounded-xl">
                <button 
                  onClick={handlePrevPage}
                  disabled={currentPage === 1}
                  className="p-1 hover:bg-white/5 rounded disabled:opacity-30 disabled:cursor-not-allowed text-gray-300"
                >
                  <ChevronLeft className="h-4 w-4" />
                </button>
                <span className="text-[11px] font-semibold min-w-16 text-center select-none text-gray-300">
                  Page {currentPage} of {activeDoc.page_count}
                </span>
                <button 
                  onClick={handleNextPage}
                  disabled={currentPage === activeDoc.page_count}
                  className="p-1 hover:bg-white/5 rounded disabled:opacity-30 disabled:cursor-not-allowed text-gray-300"
                >
                  <ChevronRight className="h-4 w-4" />
                </button>
              </div>

              {/* Close PDF View button to return to workspace hub */}
              <button 
                onClick={() => setActiveDoc(null)}
                className="text-[11px] font-semibold bg-white/5 hover:bg-white/10 text-gray-300 px-3 py-1.5 rounded-lg flex items-center space-x-1"
              >
                <X className="h-3 w-3" />
                <span>Close Document</span>
              </button>
            </div>
            
            {/* Rendered PDF Page PNG */}
            <div className="flex-1 flex items-center justify-center p-6 min-h-0 bg-[#070b13]/50">
              <div className="shadow-2xl border border-white/10 rounded-xl overflow-hidden bg-white max-h-full max-w-full flex items-center justify-center">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={api.getPageImageUrl(activeDoc.id, currentPage)}
                  alt={`Page ${currentPage}`}
                  className="max-h-[calc(100vh-140px)] max-w-full object-contain select-none"
                />
              </div>
            </div>
          </section>

        ) : (

          /* 2.2 DEFAULT NOTEBOOK WORKSPACE HUB */
          <section className="flex-1 flex flex-col border-r border-white/5 relative min-w-0 bg-[#090f1f]">
            
            {/* Workspace Hub Navigation bar */}
            <div className="p-4 border-b border-white/5 flex items-center justify-between bg-[#0b1224] shrink-0">
              <div className="flex space-x-1 bg-gray-900 p-0.5 rounded-xl border border-white/5">
                <button
                  onClick={() => setActiveTab("notes")}
                  className={`text-xs font-semibold px-4 py-1.5 rounded-lg transition-all ${
                    activeTab === "notes"
                      ? "bg-gradient-to-r from-sky-500 to-indigo-600 text-white shadow"
                      : "text-gray-400 hover:text-gray-300"
                  }`}
                >
                  Saved Notes ({notes.length})
                </button>
                <button
                  onClick={() => setActiveTab("guide")}
                  className={`text-xs font-semibold px-4 py-1.5 rounded-lg transition-all ${
                    activeTab === "guide"
                      ? "bg-gradient-to-r from-sky-500 to-indigo-600 text-white shadow"
                      : "text-gray-400 hover:text-gray-300"
                  }`}
                >
                  Notebook Guides
                </button>
              </div>

              {/* Add custom note button */}
              {activeTab === "notes" && (
                <button
                  onClick={() => handleOpenNoteEditor()}
                  className="bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-bold px-3 py-1.5 rounded-lg flex items-center space-x-1.5 transition-all"
                >
                  <Plus className="h-3.5 w-3.5" />
                  <span>Add Note</span>
                </button>
              )}
            </div>

            {/* Hub Workspace Body */}
            <div className="flex-1 overflow-y-auto p-6 min-h-0">
              
              {/* TAB A: SAVED NOTES grid */}
              {activeTab === "notes" && (
                <div>
                  {notes.length === 0 ? (
                    <div className="flex flex-col items-center justify-center text-center py-24 space-y-4">
                      <div className="p-4 bg-indigo-500/5 rounded-full border border-indigo-500/10 text-indigo-400">
                        <Pin className="h-8 w-8" />
                      </div>
                      <div>
                        <h3 className="text-sm font-bold text-white">Your Notes Board is Empty</h3>
                        <p className="text-xs text-gray-500 max-w-sm mt-1">
                          Pin important answers from the chat sidebar, or click "Add Note" to write down custom research insights.
                        </p>
                      </div>
                    </div>
                  ) : (
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      {notes.map((note) => (
                        <div
                          key={note.id}
                          onClick={() => handleOpenNoteEditor(note)}
                          className="bg-[#121b30]/60 border border-white/5 hover:border-white/10 rounded-xl p-4.5 cursor-pointer transition-all hover:scale-[1.01] hover:shadow-lg hover:shadow-black/20 flex flex-col justify-between space-y-4 group relative"
                        >
                          <div className="space-y-1.5">
                            <h4 className="text-xs font-bold text-white leading-snug truncate pr-6">{note.title}</h4>
                            <div className="text-[11px] text-gray-400 line-clamp-4 leading-relaxed whitespace-pre-wrap">
                              {note.content}
                            </div>
                          </div>
                          
                          <div className="flex items-center justify-between pt-2 border-t border-white/5 text-[9px] font-mono text-gray-500">
                            <span>Edited {new Date(note.updated_at).toLocaleDateString()}</span>
                            <button
                              onClick={(e) => handleDeleteNote(note.id, e)}
                              className="text-gray-500 hover:text-red-400 p-1 rounded opacity-0 group-hover:opacity-100 transition-opacity"
                            >
                              <Trash2 className="h-3 w-3" />
                            </button>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}

              {/* TAB B: NOTEBOOK GUIDES generation center */}
              {activeTab === "guide" && (
                <div className="space-y-6">
                  
                  {/* Select guide generator panel */}
                  {selectedDocIds.length === 0 ? (
                    <div className="flex flex-col items-center justify-center text-center py-12 text-gray-500 bg-[#0e162a]/30 border border-white/5 rounded-xl p-6">
                      <HelpCircle className="h-8 w-8 text-indigo-500/50 mb-2" />
                      <p className="text-xs">Please select one or more document sources in the left sidebar to generate guides.</p>
                    </div>
                  ) : (
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                      <button
                        onClick={() => handleGenerateGuide("briefing")}
                        className="bg-[#101930] hover:bg-[#14203e] border border-white/5 rounded-xl p-4 flex flex-col items-center text-center space-y-2 transition-all hover:scale-[1.02]"
                      >
                        <FileCheck className="h-6 w-6 text-sky-400" />
                        <span className="text-[10px] font-bold text-gray-200">Briefing Doc</span>
                      </button>
                      
                      <button
                        onClick={() => handleGenerateGuide("faq")}
                        className="bg-[#101930] hover:bg-[#14203e] border border-white/5 rounded-xl p-4 flex flex-col items-center text-center space-y-2 transition-all hover:scale-[1.02]"
                      >
                        <HelpCircle className="h-6 w-6 text-green-400" />
                        <span className="text-[10px] font-bold text-gray-200">FAQ Sheet</span>
                      </button>

                      <button
                        onClick={() => handleGenerateGuide("study_guide")}
                        className="bg-[#101930] hover:bg-[#14203e] border border-white/5 rounded-xl p-4 flex flex-col items-center text-center space-y-2 transition-all hover:scale-[1.02]"
                      >
                        <BookOpen className="h-6 w-6 text-amber-400" />
                        <span className="text-[10px] font-bold text-gray-200">Study Guide</span>
                      </button>

                      <button
                        onClick={() => handleGenerateGuide("podcast")}
                        className="bg-[#101930] hover:bg-[#14203e] border border-white/5 rounded-xl p-4 flex flex-col items-center text-center space-y-2 transition-all hover:scale-[1.02]"
                      >
                        <Headphones className="h-6 w-6 text-purple-400 animate-pulse" />
                        <span className="text-[10px] font-bold text-gray-200">Audio Overview</span>
                      </button>
                    </div>
                  )}

                  {/* Generated Guide display pane */}
                  {activeGuideType && (
                    <div className="bg-[#0f172a]/60 border border-white/5 rounded-2xl p-6 space-y-4">
                      
                      {/* Guide Display Header */}
                      <div className="flex items-center justify-between pb-3 border-b border-white/5">
                        <div className="flex items-center space-x-2">
                          <Sparkles className="h-4 w-4 text-indigo-400" />
                          <h3 className="text-xs font-bold uppercase tracking-wider text-gray-200">
                            Generated {activeGuideType.replace("_", " ")}
                          </h3>
                        </div>

                        {/* Guide Utility options (Save as note, play podcast) */}
                        <div className="flex items-center space-x-2">
                          {activeGuideType === "podcast" && guideText && !guideStreaming && (
                            <button
                              onClick={handlePlayPodcast}
                              className={`text-[10px] font-bold px-3 py-1.5 rounded-lg flex items-center space-x-1.5 transition-all ${
                                isPlayingPodcast 
                                  ? "bg-red-500/20 text-red-400 border border-red-500/30" 
                                  : "bg-purple-600 hover:bg-purple-500 text-white"
                              }`}
                            >
                              {isPlayingPodcast ? (
                                <>
                                  <Square className="h-3 w-3 fill-red-400" />
                                  <span>Stop Podcast</span>
                                </>
                              ) : (
                                <>
                                  <Play className="h-3 w-3 fill-white" />
                                  <span>Play Audio Overview</span>
                                </>
                              )}
                            </button>
                          )}
                          
                          {guideText && !guideStreaming && (
                            <button
                              onClick={handleSaveGuideAsNote}
                              className="bg-indigo-600 hover:bg-indigo-500 text-white text-[10px] font-bold px-3 py-1.5 rounded-lg flex items-center space-x-1 transition-all"
                            >
                              <Pin className="h-3 w-3" />
                              <span>Save to Notes</span>
                            </button>
                          )}
                        </div>
                      </div>

                      {/* Stream loading placeholder */}
                      {guideStreaming && !guideText && (
                        <div className="flex items-center space-x-3 text-xs text-gray-400 italic py-12 justify-center">
                          <Loader2 className="h-4 w-4 animate-spin text-indigo-400" />
                          <span>Scanning source chunks and synthesizing document...</span>
                        </div>
                      )}

                      {/* Rendered Guide content */}
                      {guideText && (
                        <div className="space-y-4">
                          
                          {/* If Podcast script, render speaking lines with host cards */}
                          {activeGuideType === "podcast" ? (
                            <div className="space-y-3.5 max-h-[500px] overflow-y-auto p-2 bg-black/20 rounded-xl">
                              {podcastScript.length === 0 ? (
                                <p className="text-xs text-gray-500 italic p-4 text-center">{guideText}</p>
                              ) : (
                                podcastScript.map((line, idx) => (
                                  <div
                                    key={idx}
                                    className={`flex flex-col space-y-1 p-2.5 rounded-xl border transition-all ${
                                      idx === currentPodcastLine
                                        ? "bg-purple-500/10 border-purple-500/30 scale-[1.005] shadow"
                                        : "bg-white/5 border-transparent opacity-85"
                                    }`}
                                  >
                                    <div className="flex items-center space-x-2">
                                      <span className={`text-[10px] uppercase font-bold px-2 py-0.5 rounded ${
                                        line.speaker === "Dan" 
                                          ? "bg-sky-500/20 text-sky-400" 
                                          : "bg-purple-500/20 text-purple-400"
                                      }`}>
                                        {line.speaker}
                                      </span>
                                      {idx === currentPodcastLine && (
                                        <Loader2 className="h-3 w-3 animate-spin text-purple-400" />
                                      )}
                                    </div>
                                    <p className="text-xs text-gray-300 leading-relaxed pl-1">{line.text}</p>
                                  </div>
                                ))
                              )}
                            </div>
                          ) : (
                            /* Standard Markdown guides */
                            <div className="prose prose-invert max-w-none max-h-[500px] overflow-y-auto p-4 bg-black/10 rounded-xl">
                              {formatMarkdown(guideText)}
                            </div>
                          )}
                        </div>
                      )}

                    </div>
                  )}

                </div>
              )}

            </div>
          </section>
        )}

        {/* 3. RIGHT SIDEBAR PANEL (Persistent RAG Chat) */}
        <section className="w-[450px] flex flex-col bg-[#0b1224] shrink-0">
          
          {/* Engine Settings Header */}
          <div className="p-4 border-b border-white/5 flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Settings className="h-4 w-4 text-indigo-400" />
              <span className="text-[10px] font-bold text-gray-400 uppercase tracking-wider">Engine Settings</span>
            </div>
            <div className="flex bg-gray-950 p-0.5 rounded-lg border border-white/5">
              <button
                onClick={() => setModelType("OpenAI")}
                className={`text-[9px] font-extrabold px-2.5 py-1 rounded-md transition-all ${
                  modelType === "OpenAI"
                    ? "bg-gradient-to-r from-sky-500 to-indigo-600 text-white shadow"
                    : "text-gray-400 hover:text-gray-300"
                }`}
              >
                OpenAI
              </button>
              <button
                onClick={() => setModelType("Local")}
                className={`text-[9px] font-extrabold px-2.5 py-1 rounded-md transition-all ${
                  modelType === "Local"
                    ? "bg-gradient-to-r from-sky-500 to-indigo-600 text-white shadow"
                    : "text-gray-400 hover:text-gray-300"
                }`}
              >
                MiniCPM (CPU)
              </button>
            </div>
          </div>

          {/* Active sources prompt */}
          <div className="px-4 py-2 bg-indigo-500/5 border-b border-white/5 flex items-center justify-between text-[10px]">
            <span className="text-gray-400">Querying context from:</span>
            <span className="font-semibold text-indigo-400 font-mono">
              {selectedDocIds.length} checked {selectedDocIds.length === 1 ? "source" : "sources"}
            </span>
          </div>

          {/* Chat Message Logs */}
          <div className="flex-1 overflow-y-auto p-4 space-y-4 min-h-0">
            {messages.length === 0 ? (
              <div className="h-full flex flex-col items-center justify-center text-center p-6 space-y-4">
                <div className="p-4 bg-indigo-500/5 rounded-full border border-indigo-500/10 text-indigo-400 animate-pulse">
                  <MessageSquare className="h-7 w-7" />
                </div>
                <div>
                  <p className="text-xs font-semibold text-white">Ask your Selected Sources</p>
                  <p className="text-[10px] text-gray-500 max-w-xs mt-1">
                    Send a query to retrieve relevant content blocks and stream a response synthesized directly from your checked PDFs.
                  </p>
                </div>
              </div>
            ) : (
              messages.map((msg) => (
                <div
                  key={msg.id}
                  className={`flex flex-col space-y-1.5 ${
                    msg.role === "user" ? "items-end" : "items-start"
                  }`}
                >
                  <span className="text-[9px] font-bold uppercase tracking-widest text-gray-500 px-1">
                    {msg.role === "user" ? "You" : "Assistant"}
                  </span>
                  
                  {/* Message bubble card */}
                  <div
                    className={`p-3 rounded-xl text-xs leading-relaxed max-w-[90%] shadow-lg border relative group ${
                      msg.role === "user"
                        ? "bg-gradient-to-r from-sky-600 to-indigo-700 text-white border-white/5"
                        : "bg-[#11192e] text-gray-100 border-white/5"
                    }`}
                  >
                    {msg.content}
                    
                    {/* Hover Pin icon on assistant replies */}
                    {msg.role === "assistant" && (
                      <button
                        onClick={() => handlePinResponse(msg.content)}
                        title="Pin response to notes board"
                        className="absolute right-2 bottom-1.5 p-1 bg-black/40 hover:bg-black/60 rounded-md text-gray-400 hover:text-white opacity-0 group-hover:opacity-100 transition-opacity"
                      >
                        <Pin className="h-3 w-3" />
                      </button>
                    )}
                  </div>
                  
                  {/* Citations / Page References */}
                  {msg.citations && msg.citations.length > 0 && (
                    <div className="flex flex-wrap gap-1.5 pt-1 px-1">
                      {msg.citations.map((cite: any, idx: number) => {
                        // Find matching document if any to navigate to page
                        const matchedDoc = documents.find(d => d.id === cite.document_id || d.filename === cite.filename);
                        return (
                          <button
                            key={cite.chunk_id || idx}
                            onClick={() => matchedDoc && handleOpenDocFromCitation(matchedDoc.id, cite.page)}
                            className="text-[9px] bg-indigo-500/10 hover:bg-indigo-500/20 text-indigo-300 font-semibold px-2 py-0.5 rounded-full border border-indigo-500/20 flex items-center space-x-1 transition-all"
                          >
                            <span className="truncate max-w-[80px]">
                              {matchedDoc ? matchedDoc.filename.slice(0, 12) + "..." : "Page"}
                            </span>
                            <span>p.{cite.page}</span>
                            <ArrowRight className="h-2 w-2" />
                          </button>
                        );
                      })}
                    </div>
                  )}
                </div>
              ))
            )}
            
            {/* Live Streaming assistant reply */}
            {streaming && currentStreamingText && (
              <div className="flex flex-col space-y-1.5 items-start">
                <span className="text-[9px] font-bold uppercase tracking-widest text-gray-500 px-1">
                  Assistant
                </span>
                <div className="p-3 rounded-xl text-xs leading-relaxed max-w-[90%] bg-[#11192e] text-gray-100 border border-white/5 shadow-lg relative group">
                  {currentStreamingText}
                  
                  <button
                    onClick={() => handlePinResponse(currentStreamingText)}
                    title="Pin response to notes board"
                    className="absolute right-2 bottom-1.5 p-1 bg-black/40 hover:bg-black/60 rounded-md text-gray-400 hover:text-white opacity-0 group-hover:opacity-100 transition-opacity"
                  >
                    <Pin className="h-3 w-3" />
                  </button>
                </div>
                
                {/* Live Citations streaming preview */}
                {activeCitations.length > 0 && (
                  <div className="flex flex-wrap gap-1.5 pt-1 px-1">
                    {activeCitations.map((cite: any, idx: number) => {
                      const matchedDoc = documents.find(d => d.id === cite.document_id || d.filename === cite.filename);
                      return (
                        <button
                          key={cite.chunk_id || idx}
                          onClick={() => matchedDoc && handleOpenDocFromCitation(matchedDoc.id, cite.page)}
                          className="text-[9px] bg-indigo-500/10 hover:bg-indigo-500/20 text-indigo-300 font-semibold px-2 py-0.5 rounded-full border border-indigo-500/20 flex items-center space-x-1"
                        >
                          <span className="truncate max-w-[80px]">
                            {matchedDoc ? matchedDoc.filename.slice(0, 12) + "..." : "Page"}
                          </span>
                          <span>p.{cite.page}</span>
                          <ArrowRight className="h-2 w-2" />
                        </button>
                      );
                    })}
                  </div>
                )}
              </div>
            )}

            {/* Retrieval / Scanning loader */}
            {streaming && !currentStreamingText && (
              <div className="flex items-center space-x-2 text-[10px] text-gray-400 italic px-2">
                <Loader2 className="h-3 w-3 animate-spin text-indigo-400" />
                <span>Searching source database...</span>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Chat input submission form */}
          <form onSubmit={handleSendMessage} className="p-4 border-t border-white/5 bg-[#0b1224]/80">
            <div className="relative flex items-center">
              <input
                type="text"
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                disabled={streaming}
                placeholder="Type your question about checked sources..."
                className="w-full bg-[#0c1222] border border-white/5 rounded-xl py-3 pl-4 pr-12 text-xs text-white placeholder-gray-500 focus:outline-none focus:border-indigo-500 transition-colors disabled:opacity-50"
              />
              <button
                type="submit"
                disabled={!inputText.trim() || streaming}
                className="absolute right-2 p-2 rounded-lg bg-gradient-to-r from-sky-500 to-indigo-600 text-white transition-all hover:scale-105 disabled:opacity-40 disabled:scale-100 disabled:cursor-not-allowed"
              >
                <Send className="h-3.5 w-3.5" />
              </button>
            </div>
          </form>
        </section>

      </main>

      {/* 4. DIALOG MODAL: Custom Note creator & editor */}
      {showNoteEditor && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
          <form 
            onSubmit={handleSaveNote}
            className="bg-[#0f172a] border border-white/10 rounded-2xl w-full max-w-lg p-6 shadow-2xl flex flex-col space-y-4"
          >
            <div className="flex items-center justify-between pb-2 border-b border-white/5">
              <h3 className="text-sm font-bold text-white">
                {noteForm.id ? "Edit Workspace Note" : "Write Custom Note"}
              </h3>
              <button 
                type="button"
                onClick={() => setShowNoteEditor(false)}
                className="text-gray-400 hover:text-white transition-colors"
              >
                <X className="h-4.5 w-4.5" />
              </button>
            </div>

            <div className="space-y-1">
              <label className="text-[10px] uppercase font-bold text-gray-400">Note Title</label>
              <input
                type="text"
                placeholder="Untitled Note"
                value={noteForm.title}
                onChange={(e) => setNoteForm(prev => ({ ...prev, title: e.target.value }))}
                className="w-full bg-[#090e1a] border border-white/5 rounded-xl px-4 py-2.5 text-xs text-white placeholder-gray-600 focus:outline-none focus:border-indigo-500 transition-colors"
              />
            </div>

            <div className="space-y-1 flex-1 flex flex-col">
              <label className="text-[10px] uppercase font-bold text-gray-400">Content</label>
              <textarea
                rows={8}
                placeholder="Write your research ideas or notes here..."
                required
                value={noteForm.content}
                onChange={(e) => setNoteForm(prev => ({ ...prev, content: e.target.value }))}
                className="w-full bg-[#090e1a] border border-white/5 rounded-xl px-4 py-2.5 text-xs text-white placeholder-gray-600 focus:outline-none focus:border-indigo-500 transition-colors resize-none flex-1"
              />
            </div>

            <div className="flex items-center justify-end space-x-2 pt-2 border-t border-white/5">
              <button
                type="button"
                onClick={() => setShowNoteEditor(false)}
                className="bg-white/5 hover:bg-white/10 text-gray-300 text-xs font-semibold px-4 py-2 rounded-xl transition-all"
              >
                Cancel
              </button>
              <button
                type="submit"
                className="bg-gradient-to-r from-sky-500 to-indigo-600 hover:from-sky-400 hover:to-indigo-500 text-white text-xs font-bold px-4 py-2 rounded-xl transition-all"
              >
                Save Note
              </button>
            </div>
          </form>
        </div>
      )}

    </div>
  );
}
