import React from 'react';
import { Plus, MessageSquare, Calendar, Clock, Search } from 'lucide-react';

const Sidebar = ({ sessions, currentSession, onSessionSelect, onNewSession }) => {
  const formatTime = (timestamp) => {
    if (!timestamp) return '';
    const date = new Date(timestamp * 1000);
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const formatDate = (timestamp) => {
    if (!timestamp) return '';
    const date = new Date(timestamp * 1000);
    return date.toLocaleDateString();
  };

  const truncateText = (text, maxLength = 60) => {
    if (!text) return '';
    return text.length > maxLength ? text.substring(0, maxLength) + '...' : text;
  };

  return (
    <div className="w-80 bg-gradient-to-b from-blue-800 to-indigo-900 text-white flex flex-col">
      {/* Header */}
      <div className="p-6 border-b border-blue-700">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-bold">Chat History</h2>
          <button
            onClick={onNewSession}
            className="p-2 bg-blue-600 hover:bg-blue-500 rounded-lg transition-colors duration-200"
            title="New Chat"
          >
            <Plus className="h-5 w-5" />
          </button>
        </div>
        
        <div className="relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-blue-300" />
          <input
            type="text"
            placeholder="Search conversations..."
            className="w-full pl-10 pr-4 py-2 bg-blue-700 border border-blue-600 rounded-lg text-white placeholder-blue-300 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          />
        </div>
      </div>

      {/* Sessions List */}
      <div className="flex-1 overflow-y-auto">
        {sessions.length === 0 ? (
          <div className="p-6 text-center text-blue-200">
            <MessageSquare className="h-12 w-12 mx-auto mb-3 opacity-50" />
            <p>No conversations yet</p>
            <p className="text-sm mt-1">Start a new chat to begin</p>
          </div>
        ) : (
          <div className="p-4 space-y-2">
            {sessions.map((session) => (
              <div
                key={session.id}
                onClick={() => onSessionSelect(session)}
                className={`sidebar-item ${currentSession?.id === session.id ? 'active' : ''}`}
              >
                <div className="flex items-start space-x-3">
                  <div className="p-2 bg-blue-600 rounded-lg mt-1">
                    <MessageSquare className="h-4 w-4" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <h3 className="font-medium text-white truncate">
                      {session.title || `Session ${session.id.slice(-8)}`}
                    </h3>
                    <p className="text-blue-200 text-sm truncate">
                      {truncateText(session.last_message || session.user_query)}
                    </p>
                    <div className="flex items-center space-x-3 mt-2 text-xs text-blue-300">
                      <div className="flex items-center space-x-1">
                        <Clock className="h-3 w-3" />
                        <span>{formatTime(session.last_activity)}</span>
                      </div>
                      <div className="flex items-center space-x-1">
                        <Calendar className="h-3 w-3" />
                        <span>{formatDate(session.last_activity)}</span>
                      </div>
                      <span>{session.message_count} messages</span>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="p-4 border-t border-blue-700">
        <div className="text-center text-blue-300 text-sm">
          <div className="flex items-center justify-center space-x-2 mb-2">
            <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
            <span>Connected to FlightOps</span>
          </div>
          <p>{sessions.length} conversations</p>
        </div>
      </div>
    </div>
  );
};

export default Sidebar;
