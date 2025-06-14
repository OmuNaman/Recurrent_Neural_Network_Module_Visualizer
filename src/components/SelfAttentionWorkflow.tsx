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
  useReactFlow
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';

import { ThemeProvider } from '@/components/ThemeProvider';
import { ThemeToggle } from '@/components/ThemeToggle';
import { Button } from '@/components/ui/button';
import { RotateCcw, Cpu } from 'lucide-react';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog';

import { WordVectorNode } from '@/components/workflow/WordVectorNode';
import { MatrixNode } from '@/components/workflow/MatrixNode';
import { CalculationNode } from '@/components/workflow/CalculationNode';
import { ActivationNode } from '@/components/workflow/ActivationNode';
import { IntroNode } from '@/components/workflow/IntroNode';

import { initialNodes as rawInitialNodes, initialEdges } from '@/utils/workflowData';

const nodeTypes = {
  wordVector: WordVectorNode,
  matrix: MatrixNode,
  calculation: CalculationNode,
  activation: ActivationNode,
  introNode: IntroNode,
};

// Define the sequence of nodes for 2 timesteps + backward propagation
const ALL_TIMESTEP_NODES = [
  'intro-rnn',
  // Forward Pass - Timestep 1
  't1_input', 't1_h_prev', 't1_w_xh', 't1_w_hh', 't1_b_h', 't1_calc_h', 't1_w_hy', 't1_b_y', 't1_calc_y', 't1_pred',
  // Forward Pass - Timestep 2  
  't2_input', 't2_h_prev', 't2_w_xh', 't2_w_hh', 't2_b_h', 't2_calc_h', 't2_w_hy', 't2_b_y', 't2_calc_y', 't2_pred',
  // Backward Pass
  'target_t1', 'target_t2', 'loss_calculation', 'grad_pred1', 'grad_pred2', 'grad_y1', 'grad_y2', 
  'grad_h2', 'grad_h1', 'grad_why', 'grad_by', 'grad_wxh', 'grad_whh', 'grad_bh'
];

function RNNWorkflowContent({ isDark, onToggleTheme }: { isDark: boolean; onToggleTheme: (isDark: boolean) => void }) {
  const [completedNodeIds, setCompletedNodeIds] = useState<Set<string>>(new Set());
  const [showIntroModal, setShowIntroModal] = useState(true);
  const reactFlowInstance = useReactFlow();

  const handleNodeComplete = useCallback((nodeId: string) => {
    setCompletedNodeIds(prev => new Set(prev).add(nodeId));
  }, []);

  const processedNodes = useMemo(() => {
    return rawInitialNodes.map(node => {
      if (node.id === 'intro-rnn') {
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

      let disabled = true;
      const isIntroCompleted = completedNodeIds.has('intro-rnn');
      
      // Timestep 1 logic
      const isT1InputCompleted = completedNodeIds.has('t1_input') && completedNodeIds.has('t1_h_prev');
      const isT1WeightsCompleted = completedNodeIds.has('t1_w_xh') && completedNodeIds.has('t1_w_hh') && completedNodeIds.has('t1_b_h');
      const isT1CalcHCompleted = completedNodeIds.has('t1_calc_h');
      const isT1OutputWeightsCompleted = completedNodeIds.has('t1_w_hy') && completedNodeIds.has('t1_b_y');
      const isT1CalcYCompleted = completedNodeIds.has('t1_calc_y');
      const isT1PredCompleted = completedNodeIds.has('t1_pred');
      
      // Timestep 2 logic
      const isT2InputCompleted = completedNodeIds.has('t2_input') && completedNodeIds.has('t2_h_prev');
      const isT2WeightsCompleted = completedNodeIds.has('t2_w_xh') && completedNodeIds.has('t2_w_hh') && completedNodeIds.has('t2_b_h');
      const isT2CalcHCompleted = completedNodeIds.has('t2_calc_h');
      const isT2OutputWeightsCompleted = completedNodeIds.has('t2_w_hy') && completedNodeIds.has('t2_b_y');
      // const isT2CalcYCompleted = completedNodeIds.has('t2_calc_y'); // Not needed for T2_pred enabling condition

      switch (node.id) {
        // Timestep 1
        case 't1_input':
        case 't1_h_prev':
        case 't1_w_xh':
        case 't1_w_hh':
        case 't1_b_h':
          disabled = !isIntroCompleted || completedNodeIds.has(node.id);
          break;
        case 't1_calc_h':
          disabled = !isT1InputCompleted || !isT1WeightsCompleted || completedNodeIds.has(node.id);
          break;
        case 't1_w_hy':
        case 't1_b_y':
          disabled = !isT1CalcHCompleted || completedNodeIds.has(node.id);
          break;
        case 't1_calc_y':
          disabled = !isT1CalcHCompleted || !isT1OutputWeightsCompleted || completedNodeIds.has(node.id);
          break;
        case 't1_pred':
          disabled = !isT1CalcYCompleted || completedNodeIds.has(node.id);
          break;
          
        // Timestep 2 (enabled only after timestep 1 is complete)
        case 't2_input':
        case 't2_h_prev':
        case 't2_w_xh':
        case 't2_w_hh':
        case 't2_b_h':
          disabled = !isT1PredCompleted || completedNodeIds.has(node.id);
          break;
        case 't2_calc_h':
          disabled = !isT2InputCompleted || !isT2WeightsCompleted || completedNodeIds.has(node.id);
          break;
        case 't2_w_hy':
        case 't2_b_y':
          disabled = !isT2CalcHCompleted || completedNodeIds.has(node.id);
          break;
        case 't2_calc_y':
          disabled = !isT2CalcHCompleted || !isT2OutputWeightsCompleted || completedNodeIds.has(node.id);
          break;
        case 't2_pred':
          disabled = !completedNodeIds.has('t2_calc_y') || completedNodeIds.has(node.id); // Depends on t2_calc_y
          break;

        // Backward Propagation - starts after forward pass completion
        case 'target_t1':
        case 'target_t2':
          disabled = !completedNodeIds.has('t2_pred') || completedNodeIds.has(node.id);
          break;
        case 'loss_calculation':
          disabled = !completedNodeIds.has('target_t1') || !completedNodeIds.has('target_t2') || completedNodeIds.has(node.id);
          break;
        case 'grad_pred1':
        case 'grad_pred2':
          disabled = !completedNodeIds.has('loss_calculation') || completedNodeIds.has(node.id);
          break;
        case 'grad_y1':
          disabled = !completedNodeIds.has('grad_pred1') || completedNodeIds.has(node.id);
          break;
        case 'grad_y2':
          disabled = !completedNodeIds.has('grad_pred2') || completedNodeIds.has(node.id);
          break;
        case 'grad_h2':
          disabled = !completedNodeIds.has('grad_y2') || completedNodeIds.has(node.id);
          break;
        case 'grad_h1':
          disabled = !completedNodeIds.has('grad_y1') || !completedNodeIds.has('grad_h2') || completedNodeIds.has(node.id);
          break;
        case 'grad_why':
        case 'grad_by':
        case 'grad_wxh':
        case 'grad_whh':
        case 'grad_bh':
          disabled = !completedNodeIds.has('grad_h1') || !completedNodeIds.has('grad_h2') || completedNodeIds.has(node.id);
          break;
          
        default:
          disabled = true; 
          break;
      }

      if (completedNodeIds.has(node.id)) {
        disabled = true;
      }
      
      return { ...node, data: { ...node.data, onComplete: handleNodeComplete, disabled: disabled } };
    });
  }, [completedNodeIds, handleNodeComplete]);

  const [nodes, setNodes, onNodesChange] = useNodesState(processedNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);

  const onConnect = useCallback((params: Connection) => setEdges((eds) => addEdge(params, eds)), [setEdges]);

  const resetWorkflow = () => {
    setCompletedNodeIds(new Set());
    setShowIntroModal(true);
    setTimeout(() => {
        reactFlowInstance.fitView({
            nodes: [{ id: 'intro-rnn' }],
            duration: 800,
            padding: 0.2
        });
    }, 100);
  };

  useEffect(() => {
    if (showIntroModal) {
      reactFlowInstance.fitView({
        nodes: [{ id: 'intro-rnn' }],
        duration: 800,
        padding: 0.2
      });
    }
  }, [showIntroModal, reactFlowInstance]);

  useEffect(() => {
    if (completedNodeIds.has('t1_pred') && !completedNodeIds.has('t2_input')) {
      setTimeout(() => {
        reactFlowInstance.fitView({
          nodes: [{ id: 't2_input' }, { id: 't2_h_prev' }, { id: 't2_w_xh' }],
          duration: 1000,
          padding: 0.3
        });
      }, 500);
    }
    
    // Navigate to backward pass after forward pass completion
    if (completedNodeIds.has('t2_pred') && !completedNodeIds.has('target_t1')) {
      setTimeout(() => {
        reactFlowInstance.fitView({
          nodes: [{ id: 'target_t1' }, { id: 'target_t2' }, { id: 'loss_calculation' }],
          duration: 1000,
          padding: 0.3
        });
      }, 500);
    }
    
    // Navigate through backward pass stages
    if (completedNodeIds.has('loss_calculation') && !completedNodeIds.has('grad_pred1')) {
      setTimeout(() => {
        reactFlowInstance.fitView({
          nodes: [{ id: 'grad_pred1' }, { id: 'grad_pred2' }, { id: 'grad_y1' }],
          duration: 1000,
          padding: 0.3
        });
      }, 500);
    }
    
    if (completedNodeIds.has('grad_h1') && !completedNodeIds.has('grad_why')) {
      setTimeout(() => {
        reactFlowInstance.fitView({
          nodes: [{ id: 'grad_why' }, { id: 'grad_wxh' }, { id: 'grad_bh' }],
          duration: 1000,
          padding: 0.3
        });
      }, 500);
    }
  }, [completedNodeIds, reactFlowInstance]);

  return (
    <div className={`h-screen w-full flex flex-col transition-colors duration-300 relative overflow-hidden ${
       isDark ? 'bg-slate-900 text-white' : 'bg-slate-100 text-slate-900'
    }`}>
      <Dialog open={showIntroModal} onOpenChange={setShowIntroModal}>
        <DialogContent className="sm:max-w-[550px] bg-background/95 backdrop-blur-lg">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2 text-2xl">
              <Cpu className="text-purple-500" />
              Welcome to the Simplified RNN Lab!
            </DialogTitle>
            <DialogDescription className="pt-2 text-base">
              Explore a complete RNN training cycle: Forward + Backward Pass.
            </DialogDescription>
          </DialogHeader>
          <div className="grid gap-4 py-4 text-sm text-muted-foreground">
            <p>We'll process the sequence <span className="font-bold text-lg text-orange-400 font-mono">"A" → "B"</span> character by character. Then calculate gradients through <strong>Backward Propagation</strong>.</p>
            <p>Notice the <strong>same weight matrices</strong> are used at each timestep, and how gradients flow backward through time.</p>
            <div className="mt-4 p-3 bg-purple-500/10 border border-purple-500/20 rounded-lg">
              <p className="text-purple-400 font-medium">🎯 Your Mission:</p>
              <p className="text-sm">Complete the forward pass, then compute gradients through backpropagation to understand how RNNs learn!</p>
            </div>
          </div>
          <DialogFooter>
            <Button onClick={() => setShowIntroModal(false)} className="w-full">
              Let's Begin! 🚀
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <div className={`shrink-0 sticky top-4 mx-4 z-50 flex items-center justify-between backdrop-blur-md shadow-lg rounded-lg p-4 transition-colors duration-300 ${
        isDark ? 'bg-slate-800/70 border-slate-700/50' : 'bg-white/80 border-slate-300/60'
      }`}>
        <h1 className="text-xl font-bold bg-gradient-to-r from-amber-500 to-red-500 bg-clip-text text-transparent">
          RNN: Forward & Backward Pass (A→B)
        </h1>
        <div className="flex items-center gap-4">
          <div className="text-sm text-muted-foreground">
            Progress: {Math.max(0, completedNodeIds.size - 1)}/{ALL_TIMESTEP_NODES.length - 1} nodes
          </div>
          <Button onClick={resetWorkflow} variant="outline" size="sm" className="flex items-center gap-2">
            <RotateCcw className="w-4 h-4" /> Reset
          </Button>
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
          fitView
        >
          <Background gap={32} size={1.5} color={isDark ? '#334155' : '#cbd5e1'} />
          <Controls />
        </ReactFlow>
      </div>
    </div>
  );
}

export function SelfAttentionWorkflow() { // Consider renaming this component if it's no longer about Self-Attention
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