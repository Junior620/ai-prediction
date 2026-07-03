"""
Example usage of the Dashboard visualization module.

This script demonstrates how to:
1. Generate prediction vs actual charts
2. Create performance dashboards
3. Generate weekly reports
4. Customize visualizations
"""

from datetime import datetime, timedelta
from src.visualization.dashboard import Dashboard
from config.settings import get_settings


def example_prediction_chart():
    """Example: Generate a prediction chart."""
    print("=" * 80)
    print("Example 1: Generating Prediction Chart")
    print("=" * 80)
    
    # Initialize dashboard
    dashboard = Dashboard()
    
    # Define date range (last 30 days)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    try:
        # Generate chart for 1-day horizon predictions
        fig = dashboard.generate_prediction_chart(
            start_date=start_date,
            end_date=end_date,
            horizon=1,
            market="ICE_London",
            save_path="prediction_chart_1day.png"
        )
        
        print(f"✓ Chart generated successfully")
        print(f"  Saved to: prediction_chart_1day.png")
        print(f"  Date range: {start_date.date()} to {end_date.date()}")
        
    except Exception as e:
        print(f"✗ Error generating chart: {str(e)}")
    
    print()


def example_multi_horizon_charts():
    """Example: Generate charts for multiple horizons."""
    print("=" * 80)
    print("Example 2: Generating Multi-Horizon Charts")
    print("=" * 80)
    
    dashboard = Dashboard()
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    horizons = [1, 7, 30]
    
    for horizon in horizons:
        try:
            fig = dashboard.generate_prediction_chart(
                start_date=start_date,
                end_date=end_date,
                horizon=horizon,
                market="ICE_London",
                save_path=f"prediction_chart_{horizon}day.png"
            )
            
            print(f"✓ {horizon}-day horizon chart generated")
            
        except Exception as e:
            print(f"✗ Error generating {horizon}-day chart: {str(e)}")
    
    print()


def example_performance_dashboard():
    """Example: Generate a performance dashboard."""
    print("=" * 80)
    print("Example 3: Generating Performance Dashboard")
    print("=" * 80)
    
    dashboard = Dashboard()
    
    # Specify model version
    model_version = "v1.0.0"
    
    try:
        fig = dashboard.generate_performance_dashboard(
            model_version=model_version,
            save_path=f"performance_dashboard_{model_version}.png"
        )
        
        print(f"✓ Performance dashboard generated for model {model_version}")
        print(f"  Saved to: performance_dashboard_{model_version}.png")
        print(f"  Includes:")
        print(f"    - Error metrics (RMSE, MAE, MAPE)")
        print(f"    - Directional accuracy gauge")
        print(f"    - Coverage rate gauge")
        print(f"    - Metrics trend over time")
        
    except Exception as e:
        print(f"✗ Error generating dashboard: {str(e)}")
    
    print()


def example_weekly_report():
    """Example: Generate a weekly performance report."""
    print("=" * 80)
    print("Example 4: Generating Weekly Report")
    print("=" * 80)
    
    dashboard = Dashboard()
    
    # Get last Monday as week start
    today = datetime.now()
    week_start = today - timedelta(days=today.weekday())
    
    model_version = "v1.0.0"
    
    try:
        report = dashboard.generate_weekly_report(
            model_version=model_version,
            week_start=week_start,
            save_path=f"weekly_report_{week_start.strftime('%Y%m%d')}.txt"
        )
        
        if "status" in report and report["status"] != "success":
            print(f"⚠ Report status: {report['status']}")
            print(f"  Message: {report.get('message', 'N/A')}")
        else:
            print(f"✓ Weekly report generated for model {model_version}")
            print(f"  Week: {report['week_start']} to {report['week_end']}")
            print(f"  Total predictions: {report['total_predictions']}")
            print(f"  Mean Absolute Error: {report['mean_absolute_error']:.2f} USD/MT")
            print(f"  Directional Accuracy: {report['directional_accuracy']:.2%}")
            print(f"  Coverage Rate: {report['coverage_rate']:.2%}")
            print(f"\n  Best day: {report['best_day']['date']} "
                  f"(Error: {report['best_day']['error']:.2f})")
            print(f"  Worst day: {report['worst_day']['date']} "
                  f"(Error: {report['worst_day']['error']:.2f})")
            print(f"\n  Recommendations:")
            for i, rec in enumerate(report['recommendations'], 1):
                print(f"    {i}. {rec}")
        
    except Exception as e:
        print(f"✗ Error generating report: {str(e)}")
    
    print()


def example_custom_visualization():
    """Example: Create custom visualization with specific parameters."""
    print("=" * 80)
    print("Example 5: Custom Visualization")
    print("=" * 80)
    
    # Initialize dashboard with custom parameters
    dashboard = Dashboard(
        figure_size=(16, 8),  # Larger figure
        dpi=150  # Higher resolution
    )
    
    # Custom date range
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 1, 31)
    
    try:
        fig = dashboard.generate_prediction_chart(
            start_date=start_date,
            end_date=end_date,
            horizon=7,
            market="ICE_London",
            save_path="custom_chart_high_res.png"
        )
        
        print(f"✓ Custom high-resolution chart generated")
        print(f"  Figure size: 16x8 inches")
        print(f"  Resolution: 150 DPI")
        print(f"  Saved to: custom_chart_high_res.png")
        
    except Exception as e:
        print(f"✗ Error generating custom chart: {str(e)}")
    
    print()


def example_batch_report_generation():
    """Example: Generate reports for multiple models."""
    print("=" * 80)
    print("Example 6: Batch Report Generation")
    print("=" * 80)
    
    dashboard = Dashboard()
    
    model_versions = ["v1.0.0", "v1.1.0", "v1.2.0"]
    
    for version in model_versions:
        try:
            # Generate performance dashboard
            fig = dashboard.generate_performance_dashboard(
                model_version=version,
                save_path=f"dashboard_{version}.png"
            )
            
            print(f"✓ Dashboard generated for {version}")
            
        except Exception as e:
            print(f"✗ Error for {version}: {str(e)}")
    
    print()


def example_market_comparison():
    """Example: Compare predictions across different markets."""
    print("=" * 80)
    print("Example 7: Market Comparison")
    print("=" * 80)
    
    dashboard = Dashboard()
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    markets = ["ICE_London", "ICE_NY"]
    
    for market in markets:
        try:
            fig = dashboard.generate_prediction_chart(
                start_date=start_date,
                end_date=end_date,
                horizon=1,
                market=market,
                save_path=f"prediction_{market}.png"
            )
            
            print(f"✓ Chart generated for {market}")
            
        except Exception as e:
            print(f"✗ Error for {market}: {str(e)}")
    
    print()


def main():
    """Run all examples."""
    print("\n")
    print("*" * 80)
    print("DASHBOARD VISUALIZATION EXAMPLES")
    print("*" * 80)
    print("\n")
    
    # Run examples
    example_prediction_chart()
    example_multi_horizon_charts()
    example_performance_dashboard()
    example_weekly_report()
    example_custom_visualization()
    example_batch_report_generation()
    example_market_comparison()
    
    print("=" * 80)
    print("All examples completed!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - prediction_chart_*.png")
    print("  - performance_dashboard_*.png")
    print("  - weekly_report_*.txt")
    print("  - custom_chart_high_res.png")
    print("  - dashboard_v*.png")
    print("\nNote: Some examples may fail if there's no data in the database.")
    print("=" * 80)


if __name__ == "__main__":
    main()
