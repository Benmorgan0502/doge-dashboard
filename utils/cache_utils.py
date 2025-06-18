import os
import time
from datetime import datetime
from config.settings import API_CONFIGS

def get_cache_freshness():
    """Calculate actual cache freshness for all datasets"""
    freshness_data = {}
    overall_status = "fresh"
    oldest_hours = 0
    
    for dataset_name, config in API_CONFIGS.items():
        cache_file = config["cache"]
        
        if os.path.exists(cache_file):
            # Get file modification time
            mod_time = os.path.getmtime(cache_file)
            hours_old = (time.time() - mod_time) / 3600
            
            # Determine status
            if hours_old < 6:
                status = "fresh"
                status_color = "green"
                status_icon = "🟢"
            elif hours_old < 12:
                status = "recent"
                status_color = "orange"
                status_icon = "🟡"
            else:
                status = "stale"
                status_color = "red"
                status_icon = "🔴"
                overall_status = "stale"
            
            freshness_data[dataset_name] = {
                "hours_old": hours_old,
                "status": status,
                "status_color": status_color,
                "status_icon": status_icon,
                "last_updated": datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d %H:%M:%S")
            }
            
            oldest_hours = max(oldest_hours, hours_old)
        else:
            freshness_data[dataset_name] = {
                "hours_old": float('inf'),
                "status": "missing",
                "status_color": "red",
                "status_icon": "❌",
                "last_updated": "Never"
            }
            overall_status = "stale"
    
    # Determine overall status
    if overall_status != "stale":
        if oldest_hours < 6:
            overall_status = "fresh"
            overall_color = "green"
            overall_icon = "🟢"
        else:
            overall_status = "recent"
            overall_color = "orange"
            overall_icon = "🟡"
    else:
        overall_color = "red"
        overall_icon = "🔴"
    
    return {
        "datasets": freshness_data,
        "overall": {
            "status": overall_status,
            "color": overall_color,
            "icon": overall_icon,
            "oldest_hours": oldest_hours
        }
    }

def get_dataset_sizes():
    """Get the actual sizes of cached datasets"""
    sizes = {}
    
    for dataset_name, config in API_CONFIGS.items():
        cache_file = config["cache"]
        
        if os.path.exists(cache_file):
            try:
                import pandas as pd
                df = pd.read_csv(cache_file)
                sizes[dataset_name] = {
                    "rows": len(df),
                    "columns": len(df.columns),
                    "file_size_mb": round(os.path.getsize(cache_file) / 1024 / 1024, 2)
                }
            except Exception as e:
                sizes[dataset_name] = {
                    "rows": 0,
                    "columns": 0,
                    "file_size_mb": 0,
                    "error": str(e)
                }
        else:
            sizes[dataset_name] = {
                "rows": 0,
                "columns": 0,
                "file_size_mb": 0,
                "error": "File not found"
            }
    
    return sizes
