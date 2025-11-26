import React from 'react';
import { User, Bot } from 'lucide-react';

const MessageBubble = ({ role, content, timestamp }) => {
  const isUser = role === 'user';
  const isSystem = role === 'system';

  if (isSystem) {
    return (
      <div className="flex justify-center my-4">
        <div className="system-message text-center">
          <div className="text-sm">{content}</div>
          <div className="text-xs mt-1 opacity-70">{timestamp}</div>
        </div>
      </div>
    );
  }

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4`}>
      <div className={`flex ${isUser ? 'flex-row-reverse' : 'flex-row'} items-start space-x-3 max-w-2xl`}>
        {/* Avatar */}
        <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
          isUser 
            ? 'bg-gradient-to-r from-blue-500 to-indigo-600' 
            : 'bg-gradient-to-r from-gray-500 to-gray-600'
        }`}>
          {isUser ? (
            <User className="h-4 w-4 text-white" />
          ) : (
            <Bot className="h-4 w-4 text-white" />
          )}
        </div>

        {/* Message Content */}
        <div className={`message-bubble ${isUser ? 'user-message' : 'assistant-message'}`}>
          <div className="whitespace-pre-wrap break-words">{content}</div>
          <div className={`text-xs mt-2 opacity-70 ${isUser ? 'text-blue-100 text-right' : 'text-gray-500'}`}>
            {timestamp}
          </div>
        </div>
      </div>
    </div>
  );
};

export default MessageBubble;
