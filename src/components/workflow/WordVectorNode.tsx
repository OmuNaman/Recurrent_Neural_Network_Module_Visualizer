// FILE: src/components/workflow/WordVectorNode.tsx
import { Handle, Position } from '@xyflow/react';
import { Card } from '@/components/ui/card';
import { useTheme } from '@/components/ThemeProvider';
import { motion } from 'framer-motion';

interface WordVectorNodeProps {
  data: {
    label: string;
    matrix: number[][]; // Should be a 1xN vector
    description?: string;
    vocabulary: string[];
  };
}

export function WordVectorNode({ data }: WordVectorNodeProps) {
  const { isDark } = useTheme();
  // We assume the matrix is a 1xN vector for this node type
  const vector = data.matrix[0];

  return (
    <Card 
      className={`min-w-[320px] transition-all duration-300 shadow-lg rounded-lg relative overflow-hidden border-2 ${
        isDark ? 'border-sky-500/50 bg-slate-800/80' : 'border-sky-400 bg-white/90'
      }`}
    >
      <div className="p-4">
        <div className="text-center mb-4">
          <h3 className={`font-semibold text-lg ${isDark ? 'text-slate-100' : 'text-slate-800'}`}>{data.label}</h3>
          {data.description && <p className={`text-sm ${isDark ? 'text-slate-400' : 'text-slate-600'}`}>{data.description}</p>}
        </div>
        
        <div className="flex justify-center">
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              className="inline-block"
            >
              <div className={`px-2 py-1 rounded-md`}>
                {/* Vocabulary Header */}
                <div className="grid gap-1 mb-1" style={{ gridTemplateColumns: `repeat(${data.vocabulary.length}, minmax(0, 1fr))` }}>
                  {data.vocabulary.map((word, i) => (
                    <div key={`word-${i}`} className={`w-12 h-6 flex items-center justify-center text-xs font-semibold rounded-md ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>
                      {word}
                    </div>
                  ))}
                </div>
                {/* Vector Values */}
                <div className="grid gap-1" style={{ gridTemplateColumns: `repeat(${vector.length}, minmax(0, 1fr))` }}>
                  {vector.map((cell, j) => {
                    const isActive = cell > 0;
                    return (
                      <div
                        key={`cell-${j}`}
                        className={`w-12 h-12 flex items-center justify-center text-sm font-mono rounded-md border transition-colors duration-150 ${
                            isActive
                            ? (isDark ? 'bg-sky-700/50 border-sky-500/80 text-sky-300' : 'bg-sky-100 border-sky-400 text-sky-700')
                            : (isDark ? 'bg-slate-700/50 border-slate-600 text-slate-200' : 'bg-slate-100 border-slate-300 text-slate-800')
                        }`}
                      >
                        {cell.toFixed(1)} 
                      </div>
                    )
                  })}
                </div>
              </div>
            </motion.div>
        </div>
      </div>
      
      <Handle type="target" position={Position.Left} className="!bg-sky-500" />
      <Handle type="source" position={Position.Right} className="!bg-sky-500" />
    </Card>
  );
}