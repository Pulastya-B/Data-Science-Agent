import React, { useState, useRef, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Send, Plus, Search, Settings, MoreHorizontal, User, Bot, ArrowLeft, Paperclip, Sparkles, Trash2, Upload, FileText, X, Loader2 } from 'lucide-react';
import { cn } from '../lib/utils';
import { Logo } from './Logo';

const API_BASE = '/api';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

interface ChatSession {
  id: string;
  title: string;
  messages: Message[];
  updatedAt: Date;
  filePath?: string;
  fileName?: string;
}

interface UploadedFile {
  path: string;
  name: string;
  size: number;
}

export const ChatInterface: React.FC<{ onBack: () => void }> = ({ onBack }) => {
  const [sessions, setSessions] = useState<ChatSession[]>([
    { id: '1', title: 'New Analysis', messages: [], updatedAt: new Date() }
  ]);
  const [activeSessionId, setActiveSessionId] = useState('1');
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [uploadedFile, setUploadedFile] = useState<UploadedFile | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  const activeSession = sessions.find(s => s.id === activeSessionId) || sessions[0];

  useEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [activeSession.messages, isTyping]);

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    setIsUploading(true);
    try {
      const formData = new FormData();
      formData.append('file', file);
      const response = await fetch(API_BASE + '/upload', { method: 'POST', body: formData });
      if (!response.ok) throw new Error('Upload failed');
      const data = await response.json();
      setUploadedFile({ path: data.file_path, name: data.filename, size: data.size });
      setSessions(prev => prev.map(s => s.id === activeSessionId ? { ...s, filePath: data.file_path, fileName: data.filename, title: data.filename.replace(/\.[^/.]+$/, '') } : s));
    } catch (error) {
      console.error('Upload error:', error);
      alert('Failed to upload file.');
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  const removeFile = () => {
    setUploadedFile(null);
    setSessions(prev => prev.map(s => s.id === activeSessionId ? { ...s, filePath: undefined, fileName: undefined } : s));
  };

  const handleSend = async () => {
    if (!input.trim() || isTyping) return;
    const userMessage: Message = { id: Date.now().toString(), role: 'user', content: input, timestamp: new Date() };
    const newMessages = [...activeSession.messages, userMessage];
    updateSession(activeSessionId, newMessages);
    setInput('');
    setIsTyping(true);
    const assistantId = (Date.now() + 1).toString();
    updateSession(activeSessionId, [...newMessages, { id: assistantId, role: 'assistant', content: '', timestamp: new Date() }]);

    try {
      const response = await fetch(API_BASE + '/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages: newMessages.map(m => ({ role: m.role, content: m.content })), file_path: activeSession.filePath || uploadedFile?.path, stream: true }),
      });
      if (!response.ok) throw new Error('HTTP error');
      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      let assistantContent = '';
      if (reader) {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          const chunk = decoder.decode(value);
          for (const line of chunk.split('\n')) {
            if (line.startsWith('data: ')) {
              try {
                const data = JSON.parse(line.slice(6));
                if (data.error) { 
                  assistantContent = 'Error: ' + data.error; 
                  break; 
                }
                // Handle regular text content
                if (data.content) { 
                  assistantContent += data.content; 
                  updateSession(activeSessionId, [...newMessages, { id: assistantId, role: 'assistant', content: assistantContent, timestamp: new Date() }]); 
                }
                // Handle HTML report as complete chunk
                if (data.html_report) {
                  assistantContent += `<html-report>${data.html_report}</html-report>\n\n`;
                  updateSession(activeSessionId, [...newMessages, { id: assistantId, role: 'assistant', content: assistantContent, timestamp: new Date() }]);
                }
              } catch {}
            }
          }
        }
      }
      if (!assistantContent) assistantContent = "I apologize, but I couldn't generate a response.";
      updateSession(activeSessionId, [...newMessages, { id: assistantId, role: 'assistant', content: assistantContent, timestamp: new Date() }]);
    } catch (error) {
      console.error("Chat Error:", error);
      updateSession(activeSessionId, [...newMessages, { id: assistantId, role: 'assistant', content: "Error connecting to server. Ensure backend is running.", timestamp: new Date() }]);
    } finally {
      setIsTyping(false);
    }
  };

  const updateSession = (id: string, messages: Message[]) => setSessions(prev => prev.map(s => s.id === id ? { ...s, messages, updatedAt: new Date() } : s));
  const createNewChat = () => { const id = Date.now().toString(); setSessions([{ id, title: 'New Analysis', messages: [], updatedAt: new Date() }, ...sessions]); setActiveSessionId(id); setUploadedFile(null); };
  const deleteSession = (e: React.MouseEvent, id: string) => { e.stopPropagation(); if (sessions.length === 1) return; setSessions(prev => prev.filter(s => s.id !== id)); if (activeSessionId === id) setActiveSessionId(sessions.find(s => s.id !== id)?.id || ''); };
  const formatSize = (bytes: number) => bytes < 1024 ? bytes + ' B' : bytes < 1024 * 1024 ? (bytes / 1024).toFixed(1) + ' KB' : (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  
  // Function to render message content with HTML reports
  const renderMessageContent = (content: string) => {
    const htmlReportRegex = /<html-report>([\s\S]*?)<\/html-report>/g;
    const parts: React.ReactNode[] = [];
    let lastIndex = 0;
    let match;
    
    while ((match = htmlReportRegex.exec(content)) !== null) {
      // Add text before the HTML report
      if (match.index > lastIndex) {
        parts.push(
          <span key={`text-${lastIndex}`}>
            {content.substring(lastIndex, match.index)}
          </span>
        );
      }
      
      // Add the HTML report in an iframe with full width
      const htmlContent = match[1];
      parts.push(
        <div key={`report-${match.index}`} className="my-4 -mx-4 border-t border-b border-white/10">
          <iframe
            srcDoc={htmlContent}
            className="w-full bg-white"
            style={{ height: '80vh', minHeight: '600px' }}
            sandbox="allow-scripts allow-same-origin"
            title="Data Quality Report"
          />
        </div>
      );
      
      lastIndex = match.index + match[0].length;
    }
    
    // Add remaining text
    if (lastIndex < content.length) {
      parts.push(
        <span key={`text-${lastIndex}`}>
          {content.substring(lastIndex)}
        </span>
      );
    }
    
    return parts.length > 0 ? parts : content;
  };

  return (
    <div className="flex h-screen w-full bg-[#050505] overflow-hidden text-white/90">
      <input type="file" ref={fileInputRef} onChange={handleFileUpload} accept=".csv,.parquet,.xlsx" className="hidden" />
      <aside className="w-[280px] hidden md:flex flex-col border-r border-white/5 bg-[#0a0a0a]/50 backdrop-blur-xl">
        <div className="p-4 flex flex-col h-full">
          <div className="flex items-center gap-3 mb-8 px-2"><Logo className="w-8 h-8" /><span className="font-bold tracking-tight text-sm uppercase">Console</span></div>
          <button onClick={createNewChat} className="w-full flex items-center gap-3 px-4 py-3 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 transition-all text-sm font-medium mb-6"><Plus className="w-4 h-4" />New Analysis</button>
          <div className="flex-1 overflow-y-auto space-y-2">
            <p className="px-3 text-[10px] uppercase tracking-widest text-white/30 font-bold mb-2">History</p>
            {sessions.map(session => (
              <div key={session.id} onClick={() => { setActiveSessionId(session.id); setUploadedFile(session.filePath ? { path: session.filePath, name: session.fileName || '', size: 0 } : null); }}
                className={cn("group flex items-center justify-between px-4 py-3 rounded-xl cursor-pointer transition-all text-sm", activeSessionId === session.id ? "bg-white/10 text-white border border-white/10" : "text-white/40 hover:text-white/70 hover:bg-white/5")}>
                <div className="flex items-center gap-2 flex-1 min-w-0">{session.fileName && <FileText className="w-3 h-3 shrink-0 text-indigo-400" />}<span className="truncate">{session.title}</span></div>
                <Trash2 onClick={(e) => deleteSession(e, session.id)} className="w-4 h-4 opacity-0 group-hover:opacity-100 hover:text-rose-400 transition-all shrink-0" />
              </div>
            ))}
          </div>
          <div className="mt-auto pt-4 border-t border-white/5 flex items-center justify-between px-2">
            <button onClick={onBack} className="p-2 hover:bg-white/5 rounded-lg text-white/40 hover:text-white"><ArrowLeft className="w-5 h-5" /></button>
            <div className="flex gap-2"><button className="p-2 hover:bg-white/5 rounded-lg text-white/40 hover:text-white"><Settings className="w-5 h-5" /></button><button className="p-2 hover:bg-white/5 rounded-lg text-white/40 hover:text-white"><User className="w-5 h-5" /></button></div>
          </div>
        </div>
      </aside>
      <main className="flex-1 flex flex-col relative bg-gradient-to-b from-[#080808] to-[#050505]">
        <header className="h-16 flex items-center justify-between px-6 border-b border-white/5 backdrop-blur-md bg-black/20 sticky top-0 z-10">
          <div className="flex items-center gap-4"><button onClick={onBack} className="md:hidden p-2 hover:bg-white/5 rounded-lg"><ArrowLeft className="w-5 h-5" /></button><div><h2 className="text-sm font-bold text-white">{activeSession.title}</h2><p className="text-[10px] text-white/30">{activeSession.messages.length} messages{activeSession.fileName && ' • ' + activeSession.fileName}</p></div></div>
          <div className="flex items-center gap-3"><button className="p-2 text-white/40 hover:text-white"><Search className="w-5 h-5" /></button><button className="p-2 text-white/40 hover:text-white"><MoreHorizontal className="w-5 h-5" /></button></div>
        </header>
        <div ref={scrollRef} className="flex-1 overflow-y-auto p-4 md:p-8 space-y-8">
          {activeSession.messages.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-center px-4">
              <motion.div initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} className="w-16 h-16 bg-gradient-to-br from-indigo-500/20 to-rose-500/20 rounded-2xl flex items-center justify-center mb-6 border border-white/10"><Sparkles className="w-8 h-8 text-indigo-400" /></motion.div>
              <h1 className="text-2xl font-extrabold text-white mb-3">Welcome, Data Scientist</h1>
              <p className="text-white/40 max-w-sm text-sm">I'm your autonomous agent with <span className="text-indigo-400 font-semibold">82 specialized tools</span> for data profiling, ML training, and visualization.</p>
              <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }} className="mt-8 w-full max-w-md">
                <button onClick={() => fileInputRef.current?.click()} disabled={isUploading} className="w-full p-6 border-2 border-dashed border-white/10 rounded-2xl hover:border-indigo-500/50 hover:bg-indigo-500/5 transition-all group">
                  {isUploading ? <div className="flex items-center justify-center gap-3"><Loader2 className="w-6 h-6 animate-spin text-indigo-400" /><span className="text-white/60">Uploading...</span></div> : <><Upload className="w-8 h-8 mx-auto mb-3 text-white/20 group-hover:text-indigo-400" /><p className="text-white/40 text-sm">Drop your dataset here or click to upload</p><p className="text-white/20 text-xs mt-1">CSV, Parquet, or Excel</p></>}
                </button>
              </motion.div>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mt-8 w-full max-w-lg">
                {["Profile my dataset", "Train a classifier", "Generate heatmap", "Feature engineering"].map(p => <button key={p} onClick={() => setInput(p)} className="text-left px-4 py-3 rounded-xl bg-white/[0.03] border border-white/5 hover:bg-white/5 text-xs text-white/60 hover:text-white">"{p}"</button>)}
              </div>
            </div>
          ) : (
            activeSession.messages.map(msg => {
              const hasHtmlReport = msg.content?.includes('<html-report>');
              return (
              <motion.div key={msg.id} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className={cn("flex w-full gap-4", msg.role === 'user' ? "flex-row-reverse" : "flex-row")}>
                <div className={cn("w-8 h-8 rounded-lg flex items-center justify-center shrink-0 border border-white/10", msg.role === 'user' ? "bg-indigo-500/20" : "bg-white/5")}>{msg.role === 'user' ? <User className="w-4 h-4" /> : <Bot className="w-4 h-4 text-indigo-400" />}</div>
                <div className={cn("p-4 rounded-2xl text-sm", 
                  hasHtmlReport ? "w-full" : "max-w-[80%] md:max-w-[70%]",
                  msg.role === 'user' ? "bg-indigo-600/20 text-indigo-50 border border-indigo-500/20" : "bg-white/[0.03] text-white/80 border border-white/5")}>
                  <div className="whitespace-pre-wrap">{renderMessageContent(msg.content || "...")}</div>
                  <div className="mt-2 text-[10px] opacity-20 font-mono">{msg.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</div>
                </div>
              </motion.div>
              );
            })
          )}
          {isTyping && activeSession.messages[activeSession.messages.length - 1]?.role === 'user' && (
            <div className="flex gap-4"><div className="w-8 h-8 rounded-lg flex items-center justify-center shrink-0 bg-white/5 border border-white/10"><Bot className="w-4 h-4 text-indigo-400" /></div><div className="bg-white/[0.03] p-4 rounded-2xl border border-white/5"><div className="flex gap-1"><span className="w-1.5 h-1.5 bg-white/20 rounded-full animate-bounce"></span><span className="w-1.5 h-1.5 bg-white/20 rounded-full animate-bounce" style={{animationDelay:'-0.15s'}}></span><span className="w-1.5 h-1.5 bg-white/20 rounded-full animate-bounce" style={{animationDelay:'-0.3s'}}></span></div></div></div>
          )}
        </div>
        <div className="p-4 md:p-8 pt-0">
          <div className="max-w-4xl mx-auto relative">
            <div className="absolute -top-12 left-0 right-0 flex items-center gap-2">
              {(uploadedFile || activeSession.filePath) && <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-indigo-500/10 border border-indigo-500/20 text-xs text-indigo-300"><FileText className="w-3 h-3" /><span>{uploadedFile?.name || activeSession.fileName}</span>{uploadedFile?.size ? <span className="text-indigo-400/50">({formatSize(uploadedFile.size)})</span> : null}<button onClick={removeFile} className="hover:text-rose-400"><X className="w-3 h-3" /></button></div>}
              <button onClick={() => fileInputRef.current?.click()} className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-white/[0.03] border border-white/5 text-[10px] text-white/40 hover:text-white hover:bg-white/5"><Paperclip className="w-3 h-3" />{uploadedFile ? 'Change' : 'Attach Data'}</button>
            </div>
            <div className="relative">
              <textarea value={input} onChange={e => setInput(e.target.value)} onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend(); }}} placeholder={uploadedFile ? 'Ask about ' + uploadedFile.name + '...' : 'Ask your agent anything...'} className="w-full bg-[#0d0d0d] border border-white/10 rounded-2xl p-4 pr-16 text-sm min-h-[56px] max-h-48 resize-none focus:outline-none focus:border-indigo-500/50 text-white/90 placeholder:text-white/20 shadow-2xl" />
              <button onClick={handleSend} disabled={!input.trim() || isTyping} className={cn("absolute right-3 bottom-3 p-2.5 rounded-xl transition-all", input.trim() && !isTyping ? "bg-white text-black hover:scale-105" : "bg-white/5 text-white/20 cursor-not-allowed")}><Send className="w-4 h-4" /></button>
            </div>
            <p className="text-center mt-3 text-[10px] text-white/20">Data Science Agent v3.0 | 82 Tools | Powered by Groq + Gemini</p>
          </div>
        </div>
      </main>
    </div>
  );
};
