import React, { useState } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  AreaChart,
  Area,
  BarChart,
  Bar
} from 'recharts';

const ChartsPanel = ({ sensorData }) => {
  const [selectedChart, setSelectedChart] = useState('vibration');
  const [timeRange, setTimeRange] = useState('all');

  // Filter data based on time range
  const getFilteredData = () => {
    if (!sensorData || sensorData.length === 0) return [];
    
    let filtered = [...sensorData].reverse(); // Reverse to show chronological order
    
    const now = new Date();
    switch (timeRange) {
      case '1h':
        const oneHourAgo = new Date(now.getTime() - 60 * 60 * 1000);
        filtered = filtered.filter(d => new Date(d.timestamp) >= oneHourAgo);
        break;
      case '6h':
        const sixHoursAgo = new Date(now.getTime() - 6 * 60 * 60 * 1000);
        filtered = filtered.filter(d => new Date(d.timestamp) >= sixHoursAgo);
        break;
      case '24h':
        const twentyFourHoursAgo = new Date(now.getTime() - 24 * 60 * 60 * 1000);
        filtered = filtered.filter(d => new Date(d.timestamp) >= twentyFourHoursAgo);
        break;
      default:
        // 'all' - use all data
        break;
    }
    
    return filtered.map((d, index) => ({
      ...d,
      time: new Date(d.timestamp).toLocaleTimeString([], { 
        hour: '2-digit', 
        minute: '2-digit' 
      }),
      index
    }));
  };

  const data = getFilteredData();

  const formatTooltipValue = (value, name) => {
    const units = {
      vibrationX: 'm/s²',
      vibrationY: 'm/s²',
      vibrationZ: 'm/s²',
      temperature: '°C',
      humidity: '%',
      pressure: 'kPa'
    };
    
    return [
      `${typeof value === 'number' ? value.toFixed(3) : value} ${units[name] || ''}`,
      name.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase())
    ];
  };

  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-white p-3 border border-gray-300 rounded-lg shadow-lg">
          <p className="font-medium text-gray-800">{`Time: ${label}`}</p>
          {payload.map((entry, index) => (
            <p key={index} style={{ color: entry.color }} className="text-sm">
              {formatTooltipValue(entry.value, entry.dataKey)[1]}: {formatTooltipValue(entry.value, entry.dataKey)[0]}
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  const renderVibrationChart = () => (
    <ResponsiveContainer width="100%" height={300}>
      <LineChart data={data} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis 
          dataKey="time" 
          tick={{ fontSize: 12 }}
          interval="preserveStartEnd"
        />
        <YAxis tick={{ fontSize: 12 }} />
        <Tooltip content={<CustomTooltip />} />
        <Legend />
        <Line 
          type="monotone" 
          dataKey="vibrationX" 
          stroke="#ef4444" 
          strokeWidth={2}
          name="Vibration X"
          dot={false}
        />
        <Line 
          type="monotone" 
          dataKey="vibrationY" 
          stroke="#3b82f6" 
          strokeWidth={2}
          name="Vibration Y"
          dot={false}
        />
        <Line 
          type="monotone" 
          dataKey="vibrationZ" 
          stroke="#10b981" 
          strokeWidth={2}
          name="Vibration Z"
          dot={false}
        />
      </LineChart>
    </ResponsiveContainer>
  );

  const renderEnvironmentalChart = () => (
    <ResponsiveContainer width="100%" height={300}>
      <AreaChart data={data} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis 
          dataKey="time" 
          tick={{ fontSize: 12 }}
          interval="preserveStartEnd"
        />
        <YAxis tick={{ fontSize: 12 }} />
        <Tooltip content={<CustomTooltip />} />
        <Legend />
        <Area 
          type="monotone" 
          dataKey="temperature" 
          stackId="1"
          stroke="#f59e0b" 
          fill="#fbbf24"
          fillOpacity={0.6}
          name="Temperature"
        />
        <Area 
          type="monotone" 
          dataKey="humidity" 
          stackId="2"
          stroke="#06b6d4" 
          fill="#67e8f9"
          fillOpacity={0.6}
          name="Humidity"
        />
      </AreaChart>
    </ResponsiveContainer>
  );

  const renderPressureChart = () => (
    <ResponsiveContainer width="100%" height={300}>
      <BarChart data={data.slice(-10)} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis 
          dataKey="time" 
          tick={{ fontSize: 12 }}
        />
        <YAxis tick={{ fontSize: 12 }} />
        <Tooltip content={<CustomTooltip />} />
        <Legend />
        <Bar 
          dataKey="pressure" 
          fill="#8b5cf6"
          name="Pressure"
        />
      </BarChart>
    </ResponsiveContainer>
  );

  const renderChart = () => {
    switch (selectedChart) {
      case 'vibration':
        return renderVibrationChart();
      case 'environmental':
        return renderEnvironmentalChart();
      case 'pressure':
        return renderPressureChart();
      default:
        return renderVibrationChart();
    }
  };

  const getChartStats = () => {
    if (!data || data.length === 0) return null;
    
    const latest = data[data.length - 1];
    const vibrationMagnitude = Math.sqrt(
      latest.vibrationX ** 2 + latest.vibrationY ** 2 + latest.vibrationZ ** 2
    );
    
    return {
      vibrationMagnitude: vibrationMagnitude.toFixed(3),
      temperature: latest.temperature.toFixed(1),
      humidity: latest.humidity.toFixed(1),
      pressure: latest.pressure.toFixed(2),
      dataPoints: data.length
    };
  };

  const stats = getChartStats();

  if (!data || data.length === 0) {
    return (
      <div className="p-8 text-center text-gray-500">
        <div className="text-4xl mb-4">📊</div>
        <h3 className="text-lg font-medium mb-2">No Sensor Data</h3>
        <p className="text-sm">
          Waiting for sensor data to display charts...
        </p>
      </div>
    );
  }

  return (
    <div className="p-4">
      {/* Chart Controls */}
      <div className="flex flex-wrap items-center justify-between mb-4 gap-4">
        <div className="flex space-x-2">
          <button
            onClick={() => setSelectedChart('vibration')}
            className={`px-3 py-2 text-sm font-medium rounded-lg transition-colors ${
              selectedChart === 'vibration' 
                ? 'bg-blue-600 text-white' 
                : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
            }`}
          >
            Vibration
          </button>
          <button
            onClick={() => setSelectedChart('environmental')}
            className={`px-3 py-2 text-sm font-medium rounded-lg transition-colors ${
              selectedChart === 'environmental' 
                ? 'bg-blue-600 text-white' 
                : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
            }`}
          >
            Environment
          </button>
          <button
            onClick={() => setSelectedChart('pressure')}
            className={`px-3 py-2 text-sm font-medium rounded-lg transition-colors ${
              selectedChart === 'pressure' 
                ? 'bg-blue-600 text-white' 
                : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
            }`}
          >
            Pressure
          </button>
        </div>

        <div className="flex space-x-2">
          <select
            value={timeRange}
            onChange={(e) => setTimeRange(e.target.value)}
            className="px-3 py-2 text-sm border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="all">All Time</option>
            <option value="1h">Last Hour</option>
            <option value="6h">Last 6 Hours</option>
            <option value="24h">Last 24 Hours</option>
          </select>
        </div>
      </div>

      {/* Current Stats */}
      {stats && (
        <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mb-4">
          <div className="bg-red-50 p-3 rounded-lg border border-red-200">
            <p className="text-xs font-medium text-red-600">Vibration Magnitude</p>
            <p className="text-lg font-bold text-red-800">{stats.vibrationMagnitude} m/s²</p>
          </div>
          <div className="bg-yellow-50 p-3 rounded-lg border border-yellow-200">
            <p className="text-xs font-medium text-yellow-600">Temperature</p>
            <p className="text-lg font-bold text-yellow-800">{stats.temperature}°C</p>
          </div>
          <div className="bg-blue-50 p-3 rounded-lg border border-blue-200">
            <p className="text-xs font-medium text-blue-600">Humidity</p>
            <p className="text-lg font-bold text-blue-800">{stats.humidity}%</p>
          </div>
          <div className="bg-purple-50 p-3 rounded-lg border border-purple-200">
            <p className="text-xs font-medium text-purple-600">Pressure</p>
            <p className="text-lg font-bold text-purple-800">{stats.pressure} kPa</p>
          </div>
          <div className="bg-gray-50 p-3 rounded-lg border border-gray-200">
            <p className="text-xs font-medium text-gray-600">Data Points</p>
            <p className="text-lg font-bold text-gray-800">{stats.dataPoints}</p>
          </div>
        </div>
      )}

      {/* Chart */}
      <div className="bg-gray-50 p-4 rounded-lg">
        {renderChart()}
      </div>
    </div>
  );
};

export default ChartsPanel;