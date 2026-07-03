"""
Dashboard and Visualization Module for the Cocoa Price Prediction System.

This module provides comprehensive visualization capabilities including:
- Prediction vs actual price charts with confidence intervals
- Performance metrics dashboard
- Market shock period highlighting
- Weekly performance reports

Requirements addressed: 11.1, 11.2, 11.3, 11.4, 11.5
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from supabase import Client, create_client

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend for server environments
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logging.warning("matplotlib not available, visualization features will be limited")

try:
    from src.models.data_models import Prediction, ModelMetrics
except ImportError:
    # Fallback for testing environments
    Prediction = None
    ModelMetrics = None

try:
    from config.settings import get_settings
except ImportError:
    # Fallback for testing environments
    get_settings = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Dashboard:
    """
    Visualization and reporting dashboard for the prediction system.
    
    This class provides methods to:
    - Generate prediction vs actual price charts
    - Display confidence interval bands
    - Show current performance metrics
    - Highlight market shock periods
    - Generate weekly performance reports
    
    Attributes:
        supabase_client: Supabase client for database operations
        figure_size: Default figure size for charts (width, height)
        dpi: Resolution for saved figures
    """
    
    def __init__(
        self,
        supabase_client: Optional[Client] = None,
        figure_size: Tuple[int, int] = (12, 6),
        dpi: int = 100
    ):
        """
        Initialize the Dashboard.
        
        Args:
            supabase_client: Optional Supabase client. If None, creates a new client.
            figure_size: Default figure size as (width, height) in inches
            dpi: Resolution for saved figures
        
        Raises:
            ImportError: If matplotlib is not available
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError(
                "matplotlib is required for visualization. "
                "Install it with: pip install matplotlib"
            )
        
        self.figure_size = figure_size
        self.dpi = dpi
        
        # Initialize Supabase client
        if supabase_client is None:
            if get_settings is None:
                raise ImportError("config.settings module not available")
            settings = get_settings()
            self.supabase_client = create_client(
                settings.supabase_url,
                settings.supabase_key
            )
        else:
            self.supabase_client = supabase_client
        
        logger.info(
            f"Dashboard initialized with figure_size={figure_size}, dpi={dpi}"
        )
    
    def generate_prediction_chart(
        self,
        start_date: datetime,
        end_date: datetime,
        horizon: int = 1,
        market: str = "ICE_London",
        save_path: Optional[str] = None
    ) -> Figure:
        """
        Generate a chart showing predicted vs actual prices with confidence intervals.
        
        Creates a visualization with:
        - Actual price line (solid blue)
        - Predicted price line (dashed orange)
        - Confidence interval bands (shaded area)
        - Market shock periods highlighted (red background)
        
        Args:
            start_date: Start date for the chart
            end_date: End date for the chart
            horizon: Prediction horizon in days (default: 1)
            market: Market identifier (default: "ICE_London")
            save_path: Optional path to save the figure
        
        Returns:
            Matplotlib Figure object
        
        Raises:
            ValueError: If no data is available for the specified period
        
        Requirements: 11.1, 11.2, 11.4
        """
        logger.info(
            f"Generating prediction chart for {market} from {start_date} to {end_date}, "
            f"horizon={horizon}"
        )
        
        # Fetch actual prices
        actual_prices = self._fetch_actual_prices(start_date, end_date, market)
        if actual_prices.empty:
            raise ValueError(
                f"No actual price data found for {market} between {start_date} and {end_date}"
            )
        
        # Fetch predictions
        predictions = self._fetch_predictions(start_date, end_date, horizon)
        if predictions.empty:
            logger.warning(
                f"No predictions found for horizon={horizon} between {start_date} and {end_date}"
            )
        
        # Detect market shocks
        shock_periods = self._detect_shock_periods(actual_prices)
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
        
        # Plot actual prices
        ax.plot(
            actual_prices['timestamp'],
            actual_prices['price'],
            label='Actual Price',
            color='#1f77b4',
            linewidth=2,
            marker='o',
            markersize=4
        )
        
        # Plot predictions if available
        if not predictions.empty:
            ax.plot(
                predictions['timestamp'],
                predictions['predicted_price'],
                label='Predicted Price',
                color='#ff7f0e',
                linewidth=2,
                linestyle='--',
                marker='s',
                markersize=4
            )
            
            # Plot confidence interval bands
            ax.fill_between(
                predictions['timestamp'],
                predictions['lower_bound'],
                predictions['upper_bound'],
                alpha=0.3,
                color='#ff7f0e',
                label='95% Confidence Interval'
            )
        
        # Highlight market shock periods
        for shock_start, shock_end in shock_periods:
            ax.axvspan(
                shock_start,
                shock_end,
                alpha=0.2,
                color='red',
                label='Market Shock' if shock_start == shock_periods[0][0] else ''
            )
        
        # Formatting
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Price (USD/MT)', fontsize=12, fontweight='bold')
        ax.set_title(
            f'Cocoa Price Prediction vs Actual - {market}\n'
            f'{start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")} '
            f'(Horizon: {horizon} day{"s" if horizon > 1 else ""})',
            fontsize=14,
            fontweight='bold'
        )
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        fig.autofmt_xdate()
        
        plt.tight_layout()
        
        # Save figure if path provided
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Chart saved to {save_path}")
        
        return fig
    
    def generate_performance_dashboard(
        self,
        model_version: str,
        save_path: Optional[str] = None
    ) -> Figure:
        """
        Generate a dashboard showing current performance metrics.
        
        Creates a multi-panel dashboard with:
        - RMSE, MAE, MAPE metrics (bar chart)
        - Directional accuracy gauge
        - Coverage rate gauge
        - Metrics trend over time (line chart)
        
        Args:
            model_version: Version identifier of the model
            save_path: Optional path to save the figure
        
        Returns:
            Matplotlib Figure object
        
        Raises:
            ValueError: If no metrics are available for the model
        
        Requirements: 11.3
        """
        logger.info(f"Generating performance dashboard for model {model_version}")
        
        # Fetch metrics
        metrics_history = self._fetch_metrics_history(model_version, limit=30)
        if not metrics_history:
            raise ValueError(f"No metrics found for model {model_version}")
        
        # Get latest metrics
        latest_metrics = metrics_history[0]
        
        # Create figure with subplots
        fig = plt.figure(figsize=(14, 10), dpi=self.dpi)
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # 1. Error Metrics Bar Chart
        ax1 = fig.add_subplot(gs[0, :])
        error_metrics = {
            'RMSE': latest_metrics['rmse'],
            'MAE': latest_metrics['mae'],
            'MAPE': latest_metrics['mape']
        }
        bars = ax1.bar(
            error_metrics.keys(),
            error_metrics.values(),
            color=['#1f77b4', '#ff7f0e', '#2ca02c'],
            alpha=0.7
        )
        ax1.set_ylabel('Error Value', fontsize=11, fontweight='bold')
        ax1.set_title(
            f'Error Metrics - Model {model_version}',
            fontsize=13,
            fontweight='bold'
        )
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.,
                height,
                f'{height:.4f}',
                ha='center',
                va='bottom',
                fontsize=10
            )
        
        # 2. Directional Accuracy Gauge
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_gauge(
            ax2,
            latest_metrics['directional_accuracy'],
            'Directional Accuracy',
            color='#2ca02c'
        )
        
        # 3. Coverage Rate Gauge
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_gauge(
            ax3,
            latest_metrics['coverage_rate'],
            'Coverage Rate',
            color='#9467bd'
        )
        
        # 4. Metrics Trend Over Time
        ax4 = fig.add_subplot(gs[2, :])
        
        # Prepare data for trend
        timestamps = [m['timestamp'] for m in reversed(metrics_history)]
        rmse_values = [m['rmse'] for m in reversed(metrics_history)]
        mae_values = [m['mae'] for m in reversed(metrics_history)]
        
        ax4.plot(timestamps, rmse_values, label='RMSE', marker='o', linewidth=2)
        ax4.plot(timestamps, mae_values, label='MAE', marker='s', linewidth=2)
        
        ax4.set_xlabel('Date', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Error Value', fontsize=11, fontweight='bold')
        ax4.set_title('Metrics Trend Over Time', fontsize=13, fontweight='bold')
        ax4.legend(loc='best', fontsize=10)
        ax4.grid(True, alpha=0.3)
        ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        fig.autofmt_xdate()
        
        # Add overall title
        fig.suptitle(
            f'Performance Dashboard - Model {model_version}',
            fontsize=16,
            fontweight='bold',
            y=0.98
        )
        
        # Save figure if path provided
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Dashboard saved to {save_path}")
        
        return fig
    
    def generate_weekly_report(
        self,
        model_version: str,
        week_start: Optional[datetime] = None,
        save_path: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Generate a weekly performance report comparing predictions to actual outcomes.
        
        Creates a comprehensive report including:
        - Summary statistics (mean error, accuracy)
        - Daily prediction accuracy breakdown
        - Best and worst prediction days
        - Recommendations for improvement
        
        Args:
            model_version: Version identifier of the model
            week_start: Start date of the week (defaults to last Monday)
            save_path: Optional path to save the report as text file
        
        Returns:
            Dictionary containing report data and statistics
        
        Requirements: 11.5
        """
        # Default to last Monday if not specified
        if week_start is None:
            today = datetime.now()
            week_start = today - timedelta(days=today.weekday())
        
        week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)
        week_end = week_start + timedelta(days=7)
        
        logger.info(
            f"Generating weekly report for model {model_version} "
            f"from {week_start} to {week_end}"
        )
        
        # Fetch predictions and actuals for the week
        predictions = self._fetch_predictions(week_start, week_end, horizon=1)
        actual_prices = self._fetch_actual_prices(
            week_start,
            week_end,
            market="ICE_London"
        )
        
        if predictions.empty or actual_prices.empty:
            logger.warning("Insufficient data for weekly report")
            return {
                "status": "insufficient_data",
                "message": "Not enough data available for the specified week"
            }
        
        # Merge predictions with actuals
        merged = pd.merge(
            predictions,
            actual_prices,
            left_on='timestamp',
            right_on='timestamp',
            how='inner',
            suffixes=('_pred', '_actual')
        )
        
        if merged.empty:
            logger.warning("No matching predictions and actuals found")
            return {
                "status": "no_matches",
                "message": "No matching predictions and actual prices found"
            }
        
        # Calculate errors
        merged['error'] = merged['predicted_price'] - merged['price']
        merged['abs_error'] = np.abs(merged['error'])
        merged['pct_error'] = (merged['abs_error'] / merged['price']) * 100
        merged['within_ci'] = (
            (merged['price'] >= merged['lower_bound']) &
            (merged['price'] <= merged['upper_bound'])
        )
        
        # Calculate directional accuracy
        if len(merged) >= 2:
            actual_direction = np.diff(merged['price'].values) > 0
            pred_direction = np.diff(merged['predicted_price'].values) > 0
            directional_accuracy = np.mean(actual_direction == pred_direction)
        else:
            directional_accuracy = 0.0
        
        # Summary statistics
        summary = {
            "week_start": week_start.strftime("%Y-%m-%d"),
            "week_end": week_end.strftime("%Y-%m-%d"),
            "model_version": model_version,
            "total_predictions": len(merged),
            "mean_absolute_error": float(merged['abs_error'].mean()),
            "mean_percentage_error": float(merged['pct_error'].mean()),
            "rmse": float(np.sqrt(np.mean(merged['error'] ** 2))),
            "directional_accuracy": float(directional_accuracy),
            "coverage_rate": float(merged['within_ci'].mean()),
            "best_day": {
                "date": merged.loc[merged['abs_error'].idxmin(), 'timestamp'].strftime("%Y-%m-%d"),
                "error": float(merged['abs_error'].min())
            },
            "worst_day": {
                "date": merged.loc[merged['abs_error'].idxmax(), 'timestamp'].strftime("%Y-%m-%d"),
                "error": float(merged['abs_error'].max())
            }
        }
        
        # Daily breakdown
        daily_breakdown = []
        for _, row in merged.iterrows():
            daily_breakdown.append({
                "date": row['timestamp'].strftime("%Y-%m-%d"),
                "actual_price": float(row['price']),
                "predicted_price": float(row['predicted_price']),
                "error": float(row['error']),
                "abs_error": float(row['abs_error']),
                "pct_error": float(row['pct_error']),
                "within_ci": bool(row['within_ci'])
            })
        
        summary["daily_breakdown"] = daily_breakdown
        
        # Generate recommendations
        recommendations = self._generate_recommendations(summary)
        summary["recommendations"] = recommendations
        
        # Save report if path provided
        if save_path:
            self._save_text_report(summary, save_path)
            logger.info(f"Weekly report saved to {save_path}")
        
        logger.info(
            f"Weekly report generated: MAE={summary['mean_absolute_error']:.2f}, "
            f"Coverage={summary['coverage_rate']:.2%}"
        )
        
        return summary
    
    def _fetch_actual_prices(
        self,
        start_date: datetime,
        end_date: datetime,
        market: str
    ) -> pd.DataFrame:
        """Fetch actual price data from database."""
        try:
            response = (
                self.supabase_client
                .table("price_data")
                .select("timestamp, price")
                .eq("market", market)
                .gte("timestamp", start_date.isoformat())
                .lte("timestamp", end_date.isoformat())
                .order("timestamp")
                .execute()
            )
            
            if not response.data:
                return pd.DataFrame()
            
            df = pd.DataFrame(response.data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['price'] = df['price'].astype(float)
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch actual prices: {str(e)}")
            return pd.DataFrame()
    
    def _fetch_predictions(
        self,
        start_date: datetime,
        end_date: datetime,
        horizon: int
    ) -> pd.DataFrame:
        """Fetch predictions from database."""
        try:
            response = (
                self.supabase_client
                .table("predictions")
                .select("created_at, predicted_price, lower_bound, upper_bound")
                .eq("horizon", horizon)
                .gte("created_at", start_date.isoformat())
                .lte("created_at", end_date.isoformat())
                .order("created_at")
                .execute()
            )
            
            if not response.data:
                return pd.DataFrame()
            
            df = pd.DataFrame(response.data)
            df['timestamp'] = pd.to_datetime(df['created_at'])
            df['predicted_price'] = df['predicted_price'].astype(float)
            df['lower_bound'] = df['lower_bound'].astype(float)
            df['upper_bound'] = df['upper_bound'].astype(float)
            df = df.drop('created_at', axis=1)
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch predictions: {str(e)}")
            return pd.DataFrame()
    
    def _detect_shock_periods(
        self,
        price_data: pd.DataFrame,
        threshold: float = 0.05
    ) -> List[Tuple[datetime, datetime]]:
        """
        Detect market shock periods (>5% daily price change).
        
        Returns list of (start_date, end_date) tuples for shock periods.
        """
        if len(price_data) < 2:
            return []
        
        # Calculate daily percentage changes
        price_data = price_data.sort_values('timestamp')
        price_data['pct_change'] = price_data['price'].pct_change().abs()
        
        # Identify shock days
        shocks = price_data[price_data['pct_change'] > threshold]
        
        if shocks.empty:
            return []
        
        # Group consecutive shock days
        shock_periods = []
        current_start = None
        current_end = None
        
        for timestamp in shocks['timestamp']:
            if current_start is None:
                current_start = timestamp
                current_end = timestamp
            elif (timestamp - current_end).days <= 1:
                current_end = timestamp
            else:
                shock_periods.append((current_start, current_end))
                current_start = timestamp
                current_end = timestamp
        
        # Add last period
        if current_start is not None:
            shock_periods.append((current_start, current_end))
        
        return shock_periods
    
    def _fetch_metrics_history(
        self,
        model_version: str,
        limit: int = 30
    ) -> List[Dict[str, any]]:
        """Fetch metrics history from database."""
        try:
            response = (
                self.supabase_client
                .table("model_metrics")
                .select("*")
                .eq("model_version", model_version)
                .order("created_at", desc=True)
                .limit(limit)
                .execute()
            )
            
            if not response.data:
                return []
            
            metrics_list = []
            for row in response.data:
                metrics_list.append({
                    "rmse": float(row["rmse"]),
                    "mae": float(row["mae"]),
                    "mape": float(row["mape"]),
                    "directional_accuracy": float(row["directional_accuracy"]),
                    "coverage_rate": float(row["coverage_rate"]),
                    "mean_interval_width": float(row["mean_interval_width"]),
                    "timestamp": datetime.fromisoformat(row["created_at"])
                })
            
            return metrics_list
            
        except Exception as e:
            logger.error(f"Failed to fetch metrics history: {str(e)}")
            return []
    
    def _plot_gauge(
        self,
        ax,
        value: float,
        title: str,
        color: str = '#2ca02c'
    ) -> None:
        """Plot a gauge chart for a metric value between 0 and 1."""
        # Create gauge background
        ax.barh(0, 1, height=0.3, color='lightgray', alpha=0.3)
        
        # Create gauge fill
        ax.barh(0, value, height=0.3, color=color, alpha=0.7)
        
        # Add value text
        ax.text(
            0.5, 0,
            f'{value:.2%}',
            ha='center',
            va='center',
            fontsize=16,
            fontweight='bold'
        )
        
        # Formatting
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, 0.5)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
    
    def _generate_recommendations(self, summary: Dict[str, any]) -> List[str]:
        """Generate recommendations based on weekly performance."""
        recommendations = []
        
        # Check MAE
        if summary['mean_absolute_error'] > 100:
            recommendations.append(
                "High mean absolute error detected. Consider retraining the model "
                "with more recent data or adjusting hyperparameters."
            )
        
        # Check directional accuracy
        if summary['directional_accuracy'] < 0.6:
            recommendations.append(
                "Low directional accuracy. Review the residual prediction model "
                "and consider adding more econometric features."
            )
        
        # Check coverage rate
        if summary['coverage_rate'] < 0.90:
            recommendations.append(
                "Coverage rate below 90%. Consider widening confidence intervals "
                "or improving uncertainty estimation."
            )
        elif summary['coverage_rate'] > 0.98:
            recommendations.append(
                "Coverage rate very high. Confidence intervals may be too wide. "
                "Consider tightening them for more actionable predictions."
            )
        
        # Check percentage error
        if summary['mean_percentage_error'] > 5.0:
            recommendations.append(
                "Mean percentage error above 5%. Review data quality and "
                "consider incorporating sentiment analysis more heavily."
            )
        
        # If performance is good
        if not recommendations:
            recommendations.append(
                "Model performance is within acceptable ranges. Continue monitoring."
            )
        
        return recommendations
    
    def _save_text_report(self, summary: Dict[str, any], save_path: str) -> None:
        """Save weekly report as formatted text file."""
        with open(save_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("WEEKLY PERFORMANCE REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Report Period: {summary['week_start']} to {summary['week_end']}\n")
            f.write(f"Model Version: {summary['model_version']}\n")
            f.write(f"Total Predictions: {summary['total_predictions']}\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("SUMMARY STATISTICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Mean Absolute Error: {summary['mean_absolute_error']:.2f} USD/MT\n")
            f.write(f"Mean Percentage Error: {summary['mean_percentage_error']:.2f}%\n")
            f.write(f"RMSE: {summary['rmse']:.2f} USD/MT\n")
            f.write(f"Directional Accuracy: {summary['directional_accuracy']:.2%}\n")
            f.write(f"Coverage Rate: {summary['coverage_rate']:.2%}\n\n")
            
            f.write(f"Best Day: {summary['best_day']['date']} "
                   f"(Error: {summary['best_day']['error']:.2f} USD/MT)\n")
            f.write(f"Worst Day: {summary['worst_day']['date']} "
                   f"(Error: {summary['worst_day']['error']:.2f} USD/MT)\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("DAILY BREAKDOWN\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Date':<12} {'Actual':<10} {'Predicted':<10} {'Error':<10} "
                   f"{'% Error':<10} {'In CI':<8}\n")
            f.write("-" * 80 + "\n")
            
            for day in summary['daily_breakdown']:
                f.write(
                    f"{day['date']:<12} "
                    f"{day['actual_price']:<10.2f} "
                    f"{day['predicted_price']:<10.2f} "
                    f"{day['abs_error']:<10.2f} "
                    f"{day['pct_error']:<10.2f} "
                    f"{'Yes' if day['within_ci'] else 'No':<8}\n"
                )
            
            f.write("\n" + "-" * 80 + "\n")
            f.write("RECOMMENDATIONS\n")
            f.write("-" * 80 + "\n")
            for i, rec in enumerate(summary['recommendations'], 1):
                f.write(f"{i}. {rec}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"Report generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n")
