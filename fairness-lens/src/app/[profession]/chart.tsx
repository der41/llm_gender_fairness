// src/app/[profession]/chart.tsx
'use client';

import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend } from 'recharts';

// 1. Define the matching colors constant
const COLORS: Record<string, string> = {
  Male: '#2B7FFF',    // Blue
  Female: '#F7369A',  // Pink
  Neutral: '#D1D1D1', // Gray
};

// 2. Custom Tooltip to show the clean original numbers
const CustomTooltip = ({ active, payload }: any) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload;
    return (
      <div className="bg-white p-3 border border-slate-100 shadow-xl rounded-xl">
        <p className="font-bold mb-1" style={{ color: COLORS[data.name] }}>
          {data.name}
        </p>
        <p className="text-sm text-slate-500">
          Count: <span className="font-mono font-bold text-slate-900">
            {/* Fallback to value if original is missing */}
            {data.original !== undefined ? data.original : data.value}
          </span>
        </p>
      </div>
    );
  }
  return null;
};

export function ChartComponent({ data }: { data: any[] }) {
  if (!data || data.length === 0) return <div className="flex items-center justify-center h-full text-slate-400">No Data</div>;

  return (
    <ResponsiveContainer width="100%" height="100%">
      <PieChart>
        <Pie
          data={data}
          cx="50%"
          cy="50%"
          innerRadius={60}
          outerRadius={80}
          paddingAngle={5}
          dataKey="value"
        >
          {data.map((entry, index) => (
            <Cell 
              key={`cell-${index}`} 
              // 3. Force the color based on the name
              fill={COLORS[entry.name] || entry.fill} 
              strokeWidth={0} 
            />
          ))}
        </Pie>
        <Tooltip content={<CustomTooltip />} />
        <Legend verticalAlign="bottom" height={36} iconType="circle"/>
      </PieChart>
    </ResponsiveContainer>
  );
}