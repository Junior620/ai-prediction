import { cn } from '@/lib/utils';
import { LucideIcon } from 'lucide-react';

interface MetricCardProps {
  title: string;
  value: string;
  delta?: string;
  icon?: LucideIcon;
  trend?: 'up' | 'down' | 'neutral';
  className?: string;
}

export function MetricCard({ title, value, delta, icon: Icon, trend, className }: MetricCardProps) {
  const trendColors = {
    up: 'text-green-600',
    down: 'text-red-600',
    neutral: 'text-blue-600',
  };

  return (
    <div className={cn('bg-white dark:bg-gray-800 rounded-lg shadow-md p-6', className)}>
      <div className="flex items-center justify-between mb-2">
        <p className="text-sm font-medium text-gray-600 dark:text-gray-400">{title}</p>
        {Icon && <Icon className="h-5 w-5 text-gray-400" />}
      </div>
      <div className="flex items-baseline justify-between">
        <p className="text-2xl font-bold text-gray-900 dark:text-white">{value}</p>
        {delta && trend && (
          <span className={cn('text-sm font-semibold', trendColors[trend])}>
            {delta}
          </span>
        )}
      </div>
    </div>
  );
}
