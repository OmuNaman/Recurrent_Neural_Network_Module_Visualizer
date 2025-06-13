// FILE: src/components/workflow/NeuralNetworkDiagram.tsx
import { useTheme } from '@/components/ThemeProvider';
import { motion } from 'framer-motion';

interface RNNLoopDiagramProps {
  activeStep: 'h' | 'y' | null;
}

export function NeuralNetworkDiagram({ architecture, activeNodeId }: { architecture?: any, activeNodeId: string }) {
    const { isDark } = useTheme();

    const stepMap: Record<string, 'h' | 'y' | null> = {
        't1_calc_h': 'h', 't2_calc_h': 'h', 't3_calc_h': 'h',
        't1_calc_y': 'y', 't2_calc_y': 'y', 't3_calc_y': 'y',
        't1_pred': 'y', 't2_pred': 'y', 't3_pred': 'y',
    };
    const currentStep = stepMap[activeNodeId] || null;

    const theme = {
        bg: isDark ? 'bg-slate-900/50' : 'bg-slate-100',
        text: isDark ? 'text-slate-300' : 'text-slate-700',
        cell: isDark ? 'bg-slate-800' : 'bg-white',
        border: isDark ? 'stroke-slate-600' : 'stroke-slate-300',
        activePath: isDark ? 'stroke-amber-400' : 'stroke-amber-500',
        inactivePath: isDark ? 'stroke-slate-700' : 'stroke-slate-400',
        label: isDark ? 'fill-slate-400' : 'fill-slate-600',
    };

    const pathVariants = {
        inactive: { pathLength: 1, stroke: theme.inactivePath, transition: { duration: 0.3 } },
        active: { pathLength: 1, stroke: theme.activePath, transition: { duration: 0.3 } },
    };

  return (
    <div className={`w-full h-full p-4 rounded-lg flex items-center justify-center transition-colors ${theme.bg}`}>
      <svg viewBox="0 0 400 300" width="100%" height="100%">
        <defs>
          <marker id="arrow" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" className={isDark ? "fill-slate-500" : "fill-slate-400"} />
          </marker>
          <marker id="arrow-active" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" className={isDark ? "fill-amber-400" : "fill-amber-500"} />
          </marker>
        </defs>

        {/* RNN Cell */}
        <motion.rect 
            x="125" y="100" width="150" height="100" rx="10" 
            className={`${theme.cell} transition-colors`} 
            stroke={currentStep ? theme.activePath : theme.border}
            strokeWidth="2"
        />
        <text x="200" y="155" textAnchor="middle" className={`font-bold text-lg ${theme.text}`}>RNN Cell</text>

        {/* Input */}
        <motion.path d="M 50 150 H 125" variants={pathVariants} animate={currentStep === 'h' ? 'active' : 'inactive'} markerEnd={currentStep === 'h' ? "url(#arrow-active)" : "url(#arrow)"} strokeWidth="2" />
        <text x="87.5" y="135" textAnchor="middle" className={`text-sm ${theme.label}`}>x_t (Input)</text>

        {/* Recurrent Connection */}
        <motion.path d="M 200 200 Q 200 250 150 250 L 50 250 Q 0 250 0 200 L 0 100 Q 0 50 50 50 L 150 50 Q 200 50 200 100" fill="none" variants={pathVariants} animate={currentStep === 'h' ? 'active' : 'inactive'} markerEnd={currentStep === 'h' ? "url(#arrow-active)" : "url(#arrow)"} strokeWidth="2" />
        <text x="75" y="35" textAnchor="middle" className={`text-sm ${theme.label}`}>h_t-1 (Memory)</text>
        
        {/* Hidden State Output */}
        <motion.path d="M 275 150 H 350" variants={pathVariants} animate={currentStep === 'h' || currentStep === 'y' ? 'active' : 'inactive'} markerEnd={currentStep === 'h' || currentStep === 'y' ? "url(#arrow-active)" : "url(#arrow)"} strokeWidth="2" />
        <text x="312.5" y="135" textAnchor="middle" className={`text-sm ${theme.label}`}>h_t (New Memory)</text>

        {/* Final Output */}
        <motion.path d="M 200 100 V 50 H 350" fill="none" variants={pathVariants} animate={currentStep === 'y' ? 'active' : 'inactive'} markerEnd={currentStep === 'y' ? "url(#arrow-active)" : "url(#arrow)"} strokeWidth="2" />
        <text x="312.5" y="35" textAnchor="middle" className={`text-sm ${theme.label}`}>ŷ_t (Prediction)</text>
      </svg>
    </div>
  );
}