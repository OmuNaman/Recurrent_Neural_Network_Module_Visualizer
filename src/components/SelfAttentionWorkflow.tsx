// FILE: src/components/SelfAttentionWorkflow.tsx
import { useState, useCallback, useMemo, useEffect } from 'react';
import {
  ReactFlow,
  addEdge,
  Controls,
  Background,
  useNodesState,
  useEdgesState,
  Connection,
  ReactFlowProvider,
  useReactFlow // Import useReactFlow for fitView
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';

import { ThemeProvider } from '@/components/ThemeProvider';
import { ThemeToggle } from '@/components/ThemeToggle';
import { Button } from '@/components/ui/button';
import { RotateCcw, Cpu } from 'lucide-react';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog'; // Import Dialog components

// Import all custom node types
import { WordVectorNode } from '@/components/workflow/WordVectorNode';
import { MatrixNode } from '@/components/workflow/MatrixNode';
import { CalculationNode } from '@/components/workflow/CalculationNode';
import { ActivationNode } from '@/components/workflow/ActivationNode'; // ADD THIS IMPORT
import { IntroNode } from '@/components/workflow/IntroNode';             // ADD THIS IMPORT

import { initialNodes as rawInitialNodes, initialEdges } from '@/utils/workflowData';

// Define all custom node types used in the workflow
const nodeTypes = {
  wordVector: WordVectorNode,
  matrix: MatrixNode,
  calculation: CalculationNode,
  activation: ActivationNode, // ADD THIS LINE
  introNode: IntroNode,       // ADD THIS LINE
};

// Define the sequence of nodes for the first timestep, excluding static nodes
const TIMESTEP_1_NODES = [
  'intro-rnn',
  't1_input',
  't1_h_prev',
  't1_calc_h',
  't1_calc_y',
  't1_pred',
];

function RNNWorkflowContent({ isDark, onToggleTheme }: { isDark: boolean; onToggleTheme: (isDark: boolean) => void }) {
  const [completedNodeIds, setCompletedNodeIds] = useState<Set<string>>(new Set());
  const [showIntroModal, setShowIntroModal] = useState(true); // State to control the intro modal
  const reactFlowInstance = useReactFlow(); // Get React Flow instance for fitting view

  const handleNodeComplete = useCallback((nodeId: string) => {
    setCompletedNodeIds(prev => new Set(prev).add(nodeId));
  }, []);

  const processedNodes = useMemo(() => {
    // These nodes are always enabled because they are static inputs/weights
    const alwaysEnabledNodes = ['w_xh', 'w_hh', 'w_hy', 'b_h', 'b_y'];

    return rawInitialNodes.map(node => {
      // Always enable static weight/bias nodes
      if (alwaysEnabledNodes.includes(node.id)) {
        return { ...node, data: { ...node.data, disabled: false, onComplete: handleNodeComplete } };
      }

      // Handle the intro node separately
      if (node.id === 'intro-rnn') {
        // Intro node is only active if not yet completed and no other nodes are enabled yet
        const isIntroActive = !completedNodeIds.has('intro-rnn') && completedNodeIds.size === 0;
        return { 
          ...node, 
          data: { 
            ...node.data, 
            onComplete: handleNodeComplete, 
            disabled: !isIntroActive 
          } 
        };
      }

      // Logic for Timestep 1 nodes
      let disabled = true;
      const isIntroCompleted = completedNodeIds.has('intro-rnn');
      const isInputCompleted = completedNodeIds.has('t1_input') && completedNodeIds.has('t1_h_prev');
      const isCalcHCompleted = completedNodeIds.has('t1_calc_h');
      const isCalcYCompleted = completedNodeIds.has('t1_calc_y');
      const isPredCompleted = completedNodeIds.has('t1_pred');

      switch (node.id) {
        case 't1_input':
        case 't1_h_prev':
          disabled = !isIntroCompleted || completedNodeIds.has(node.id); // Enable once intro is done
          break;
        case 't1_calc_h':
          disabled = !isInputCompleted || completedNodeIds.has(node.id);
          break;
        case 't1_calc_y':
          disabled = !isCalcHCompleted || completedNodeIds.has(node.id);
          break;
        case 't1_pred':
          disabled = !isCalcYCompleted || completedNodeIds.has(node.id);
          break;
        default:
          disabled = true; // All other future nodes are disabled for now
          break;
      }

      // Keep nodes disabled if they are already completed, or if they are inputs to a completed node
      // This prevents users from re-editing completed sections
      if (completedNodeIds.has(node.id)) {
        disabled = true;
      }
      
      return { ...node, data: { ...node.data, onComplete: handleNodeComplete, disabled: disabled } };
    });
  }, [completedNodeIds, handleNodeComplete]);

  const [nodes, setNodes, onNodesChange] = useNodesState(processedNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);

  const onConnect = useCallback((params: Connection) => setEdges((eds) => addEdge(params, eds)), [setEdges]);

  // Reset function to clear all progress and re-show modal
  const resetWorkflow = () => {
    setCompletedNodeIds(new Set());
    setShowIntroModal(true);
    // Fit view to intro node after reset
    setTimeout(() => {
        reactFlowInstance.fitView({
            nodes: [{ id: 'intro-rnn' }],
            duration: 800,
            padding: 0.2
        });
    }, 100);
  };

  // Effect to fit view to the intro node on initial load
  useEffect(() => {
    if (showIntroModal) {
      reactFlowInstance.fitView({
        nodes: [{ id: 'intro-rnn' }],
        duration: 800,
        padding: 0.2
      });
    }
  }, [showIntroModal, reactFlowInstance]);


  return (
    <div className={`h-screen w-full flex flex-col transition-colors duration-300 relative overflow-hidden ${
       isDark ? 'bg-slate-900 text-white' : 'bg-slate-100 text-slate-900'
    }`}>
      {/* Intro Modal */}
      <Dialog open={showIntroModal} onOpenChange={setShowIntroModal}>
        <DialogContent className="sm:max-w-[550px] bg-background/95 backdrop-blur-lg">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2 text-2xl">
              <Cpu className="text-purple-500" />
              Welcome to the RNN Time-Traveler Lab!
            </DialogTitle>
            <DialogDescription className="pt-2 text-base">
              You're about to explore the fundamental mechanism of Recurrent Neural Networks.
            </DialogDescription>
          </DialogHeader>
          <div className="grid gap-4 py-4 text-sm text-muted-foreground">
            <p>We'll process the sequence <span className="font-bold text-lg text-orange-400 font-mono">"RNN!"</span> character by character. At each step, you'll calculate the network's internal **Hidden State** (its memory) and its **Prediction** for the next character.</p>
            <p>Pay close attention to how the hidden state from one step is passed as input to the next!</p>
          </div>
          <DialogFooter>
            <Button onClick={() => setShowIntroModal(false)} className="w-full">
              Let's Start!
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Header */}
      <div className={`shrink-0 sticky top-4 mx-4 z-50 flex items-center justify-between backdrop-blur-md shadow-lg rounded-lg p-4 transition-colors duration-300 ${
        isDark ? 'bg-slate-800/70 border-slate-700/50' : 'bg-white/80 border-slate-300/60'
      }`}>
        <h1 className="text-xl font-bold bg-gradient-to-r from-amber-500 to-red-500 bg-clip-text text-transparent">
          Recurrent Neural Network: Forward Pass
        </h1>
        <div className="flex items-center gap-4">
          <Button onClick={resetWorkflow} variant="outline" size="sm" className="flex items-center gap-2"><RotateCcw className="w-4 h-4" /> Reset</Button>
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
          nodeTypes={nodeTypes}
          fitView // Ensure fitView is enabled
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