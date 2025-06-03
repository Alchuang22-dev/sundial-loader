import re
from typing import Dict, List, Optional, Tuple
from datetime import datetime

class IoTDBSQLParser:
    """IoTDB SQL 语句解析器示例"""
    
    @staticmethod
    def parse_select_query(sql: str) -> Dict[str, str]:
        """解析 SELECT 查询语句"""
        # 简化的 SQL 解析，实际应该连接 Apache IoTDB 中的 SQL 解析器
        pattern = r"select\s+(.+?)\s+from\s+(.+?)(?:\s+where\s+(.+?))?(?:\s+limit\s+(\d+))?"
        match = re.search(pattern, sql.lower())
        
        if not match:
            raise ValueError(f"无法解析 SQL 查询: {sql}")
        
        return {
            "columns": match.group(1).strip(),
            "table": match.group(2).strip(),
            "where": match.group(3).strip() if match.group(3) else None,
            "limit": int(match.group(4)) if match.group(4) else None
        }
    
    @staticmethod
    def parse_time_condition(where_clause: str) -> Optional[Dict[str, datetime]]:
        """解析时间条件"""
        if not where_clause:
            return None
        
        # 解析时间范围条件
        time_pattern = r"time\s*>=\s*['\"]?([^'\"]+)['\"]?"
        match = re.search(time_pattern, where_clause)
        
        if match:
            time_str = match.group(1)
            try:
                start_time = datetime.fromisoformat(time_str.replace('+08:00', ''))
                return {"start_time": start_time}
            except ValueError:
                pass
        
        return None
    
    @staticmethod
    def parse_window_expression(window: str) -> Dict[str, int]:
        """解析窗口表达式"""
        if "tail(" in window:
            # 提取 tail(n) 中的数字
            match = re.search(r"tail\((\d+)\)", window)
            if match:
                return {"type": "tail", "size": int(match.group(1))}
        
        return {"type": "default", "size": 100}