import React from 'react';
import { Brain, Cpu, Type, CheckCircle, Clock } from 'lucide-react';

const StatusIndicator = ({ agentState, tokenUsage }) => {
  const { phase, progress_pct, message } = agentState; // ✅ FIXED: Use progress_pct

  const phaseConfig = {
    thinking: {
      icon: Brain,
      color: 'text-purple-600',
      bgColor: 'bg-purple-50',
      borderColor: 'border-purple-200',
      label: 'Thinking'
    },
    processing: {
      icon: Cpu,
      color: 'text-amber-600',
      bgColor: 'bg-amber-50',
      borderColor: 'border-amber-200',
      label: 'Processing'
    },
    typing: {
      icon: Type,
      color: 'text-green-600',
      bgColor: 'bg-green-50',
      borderColor: 'border-green-200',
      label: 'Generating'
    },
    finished: {
      icon: CheckCircle,
      color: 'text-blue-600',
      bgColor: 'bg-blue-50',
      borderColor: 'border-blue-200',
      label: 'Complete'
    },
    idle: {
      icon: Clock,
      color: 'text-gray-600',
      bgColor: 'bg-gray-50',
      borderColor: 'border-gray-200',
      label: 'Ready'
    }
  };

  const config = phaseConfig[phase] || phaseConfig.idle;
  const Icon = config.icon;

  const formatTokenCount = (count) => {
    if (!count) return '0';
    return count.toLocaleString();
  };

  return (
    <div className={`p-4 rounded-xl border-l-4 ${config.borderColor} ${config.bgColor} mb-4`}>
      <div className="flex items-center space-x-3">
        <Icon className={`h-5 w-5 ${config.color}`} />
        <div className="flex-1">
          <div className="flex items-center justify-between mb-2">
            <span className={`font-medium ${config.color}`}>
              {config.label} {message && `- ${message}`}
            </span>
            {progress_pct > 0 && ( // ✅ FIXED: Use progress_pct
              <span className="text-sm text-gray-500">{progress_pct}%</span>
            )}
          </div>

          {/* Progress Bar */}
          {progress_pct > 0 && ( // ✅ FIXED: Use progress_pct
            <div className="progress-bar mb-3">
              <div 
                className="progress-fill" 
                style={{ width: `${progress_pct}%` }} // ✅ FIXED: Use progress_pct
              ></div>
            </div>
          )}

          {/* Token Usage */}
          {tokenUsage.total && (
            <div className="space-y-2 mt-3">
              <div className="grid grid-cols-3 gap-4 text-sm">
                {tokenUsage.planning && (
                  <div className="text-center p-2 bg-white rounded-lg border">
                    <div className="font-semibold text-purple-600">
                      {formatTokenCount(tokenUsage.planning.total_tokens)}
                    </div>
                    <div className="text-xs text-gray-500">Planning</div>
                  </div>
                )}
                {tokenUsage.summarization && (
                  <div className="text-center p-2 bg-white rounded-lg border">
                    <div className="font-semibold text-blue-600">
                      {formatTokenCount(tokenUsage.summarization.total_tokens)}
                    </div>
                    <div className="text-xs text-gray-500">Summarization</div>
                  </div>
                )}
                {tokenUsage.total && (
                  <div className="text-center p-2 bg-white rounded-lg border">
                    <div className="font-semibold text-green-600">
                      {formatTokenCount(tokenUsage.total.total_tokens)}
                    </div>
                    <div className="text-xs text-gray-500">Total</div>
                  </div>
                )}
              </div>

              {/* Detailed Breakdown */}
              {tokenUsage.total && (
                <div className="text-xs text-gray-600 text-center">
                  Prompt: {formatTokenCount(tokenUsage.total.prompt_tokens)} • 
                  Completion: {formatTokenCount(tokenUsage.total.completion_tokens)}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default StatusIndicator;
