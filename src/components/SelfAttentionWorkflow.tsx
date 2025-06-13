// FILE: src/components/SelfAttentionWorkflow.tsx
import { useState, useCallback } from 'react';
import {
  ReactFlow,
  addEdge,
  Controls,
  Background,
  useNodesState,
  useEdgesState,
  Connection,
  ReactFlowProvider,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';

import { ThemeProvider } from '@/components/ThemeProvider';
import { ThemeToggle } from '@/components/ThemeToggle';
import { Button } from '@/components/ui/button';
import { RotateCcw } from 'lucide-react';

import { WordVectorNode } from '@/components/workflow/WordVectorNode';
import { MatrixNode } from '@/components/workflow/MatrixNode';        // <--- ADD THIS IMPORT
import { CalculationNode } from '@/components/workflow/CalculationNode'; // <--- ADD THIS IMPORT

import { initialNodes as rawInitialNodes, initialEdges } from '@/utils/workflowData';

// Define all custom node types used in the workflow
const nodeTypes = {
  wordVector: WordVectorNode,
  matrix: MatrixNode,          // <--- ADD THIS LINE
  calculation: CalculationNode, // <--- ADD THIS LINE
};

function RNNWorkflowContent({ isDark, onToggleTheme }: { isDark: boolean; onToggleTheme: (isDark: boolean) => void }) {
  const [nodes, setNodes, onNodesChange] = useNodesState(rawInitialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);

  const onConnect = useCallback((params: Connection) => setEdges((eds) => addEdge(params, eds)), [setEdges]);

  // Placeholder for reset logic for now
  const handleReset = () => {
    // In future steps, this will reset completion states etc.
    setNodes(rawInitialNodes);
    setEdges(initialEdges);
  };

  return (
    <div className={`h-screen w-full flex flex-col transition-colors duration-300 relative overflow-hidden ${
       isDark ? 'bg-slate-900 text-white' : 'bg-slate-100 text-slate-900'
    }`}>
      {/* Header */}
      <div className={`shrink-0 sticky top-4 mx-4 z-50 flex items-center justify-between backdrop-blur-md shadow-lg rounded-lg p-4 transition-colors duration-300 ${
        isDark ? 'bg-slate-800/70 border-slate-700/50' : 'bg-white/80 border-slate-300/60'
      }`}>
        <h1 className="text-xl font-bold bg-gradient-to-r from-amber-500 to-red-500 bg-clip-text text-transparent">
          Recurrent Neural Network
        </h1>
        <div className="flex items-center gap-4">
          <Button onClick={handleReset} variant="outline" size="sm" className="flex items-center gap-2"><RotateCcw className="w-4 h-4" /> Reset</Button>
          <ThemeToggle isDark={isDark} onToggle={onToggleTheme} />
        </div>
      </div>

      <div className="flex-grow w-full h-full relative">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          nodeTypes={nodeTypes} // This is where React Flow maps the 'type' to the component
          fitView
        >
          <Background gap={32} size={1.5} color={isDark ? '#334155' : '#cbd5e1'} />
          <Controls />
        </ReactFlow>
      </div>
    </div>
  );
}

// Wrapper component to provide React Flow context
export function SelfAttentionWorkflow() {
  const [isDark, setIsDark] = useState(true);
  const handleThemeToggle = (newIsDark: boolean) => setIsDark(newIsDark);
  return (
    <ReactFlowProvider>
      <ThemeProvider isDark={isDark}>
        <RNNWorkflowContent isDark={isDark} onToggleTheme={handleThemeToggle} />
      </ThemeProvider>
    </ReactFlowProvider>
  );
}