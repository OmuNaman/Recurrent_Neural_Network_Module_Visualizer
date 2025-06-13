// FILE: src/components/workflow/IntroNode.tsx
import { Handle, Position } from '@xyflow/react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { useTheme } from '@/components/ThemeProvider';
import { Lightbulb, Atom, SquareActivity } from 'lucide-react'; // Changed icon

interface IntroNodeProps {
  data: {
    label: string;
    description: string;
    task: string;
    exampleSequence: string;
    focusAreas: string[];
  };
}

export function IntroNode({ data }: IntroNodeProps) {
  const { isDark } = useTheme();

  return (
    <Card 
      className={`w-[450px] shadow-2xl rounded-lg relative transition-all duration-300 ${
        isDark ? 'bg-slate-800/80 border-purple-500/50' : 'bg-white/90 border-purple-400'
      } border-2`}
    >
      <div className="absolute -top-3 -left-3">
          <SquareActivity className={`w-8 h-8 ${isDark ? 'text-purple-400' : 'text-purple-500'}`} style={{ transform: 'rotate(-15deg)' }} />
      </div>
      <CardHeader>
        <CardTitle className={`text-xl font-bold flex items-center gap-2 ${isDark ? 'text-slate-100' : 'text-slate-800'}`}>
            <Atom className="text-purple-500" />
            {data.label}
        </CardTitle>
        <CardDescription className="pt-2 text-base">
          {data.description}
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4 text-left">
        <div className="p-3 rounded-md bg-background/50">
            <p className="text-xs font-semibold text-muted-foreground mb-1">TASK</p>
            <p className="font-semibold text-lg text-blue-400">{data.task}</p>
        </div>
        <div className="p-3 rounded-md bg-background/50">
            <p className="text-xs font-semibold text-muted-foreground mb-1">EXAMPLE SEQUENCE</p>
            <p className="font-mono text-xl tracking-wider text-orange-400 font-bold">{data.exampleSequence}</p>
        </div>
        <div className="p-3 rounded-md bg-background/50">
            <p className="text-xs font-semibold text-muted-foreground mb-1">KEY CONCEPTS TO FOCUS ON</p>
            <ul className="text-sm text-gray-400 space-y-1">
                {data.focusAreas.map((area, index) => (
                    <li key={index} className="flex items-start gap-2">
                        <span className="text-blue-400 mt-1">•</span>
                        {area}
                    </li>
                ))}
            </ul>
        </div>
      </CardContent>
      <Handle type="source" position={Position.Right} className="!bg-purple-500" />
    </Card>
  );
}