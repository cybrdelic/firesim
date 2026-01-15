import React from 'react';
import { Copy } from 'lucide-react';

interface Props {
  title: string;
  code: string;
  language: 'typescript' | 'wgsl';
}

const CodeBlock: React.FC<Props> = ({ title, code, language }) => {
  const handleCopy = () => {
    navigator.clipboard.writeText(code);
  };

  return (
    <div className="flex flex-col h-full bg-slate-900 rounded-xl overflow-hidden border border-slate-700 shadow-xl">
      <div className="flex items-center justify-between px-4 py-3 bg-slate-800 border-b border-slate-700">
        <div className="flex items-center gap-2">
          <span className={`w-3 h-3 rounded-full ${language === 'typescript' ? 'bg-blue-500' : 'bg-orange-500'}`}></span>
          <span className="text-sm font-mono font-medium text-slate-200">{title}</span>
        </div>
        <button 
          onClick={handleCopy}
          className="p-1.5 hover:bg-slate-700 rounded-md text-slate-400 transition-colors"
          title="Copy to clipboard"
        >
          <Copy size={16} />
        </button>
      </div>
      <div className="flex-grow overflow-auto p-4 custom-scrollbar">
        <pre className="font-mono text-sm leading-relaxed">
          <code className={language === 'typescript' ? 'text-blue-100' : 'text-orange-100'}>
            {code}
          </code>
        </pre>
      </div>
    </div>
  );
};

export default CodeBlock;