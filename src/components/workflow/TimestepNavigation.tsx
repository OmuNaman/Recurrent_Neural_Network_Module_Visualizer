// FILE: src/components/workflow/TimestepNavigation.tsx
import { Button } from '@/components/ui/button';
import { ArrowLeft, ArrowRight } from 'lucide-react';

interface TimestepNavigationProps {
  currentTimestep: number;
  totalTimesteps: number;
  setTimestep: (setter: (prev: number) => number) => void;
  isStepComplete: boolean;
}

export function TimestepNavigation({ currentTimestep, totalTimesteps, setTimestep, isStepComplete }: TimestepNavigationProps) {
  const canGoForward = currentTimestep < totalTimesteps - 1 && isStepComplete;
  const canGoBack = currentTimestep > 0;

  return (
    <div className="absolute bottom-8 left-1/2 -translate-x-1/2 z-20 flex items-center gap-4 p-2 rounded-lg bg-background/70 backdrop-blur-md shadow-lg border">
      <Button
        onClick={() => setTimestep(prev => Math.max(0, prev - 1))}
        disabled={!canGoBack}
        variant="outline"
        size="sm"
      >
        <ArrowLeft className="h-4 w-4 mr-2" />
        Prev Step
      </Button>
      <span className="font-mono text-sm text-muted-foreground">
        Timestep: {currentTimestep + 1} / {totalTimesteps}
      </span>
      <Button
        onClick={() => setTimestep(prev => Math.min(totalTimesteps - 1, prev + 1))}
        disabled={!canGoForward}
        variant="outline"
        size="sm"
      >
        Next Step
        <ArrowRight className="h-4 w-4 ml-2" />
      </Button>
    </div>
  );
}