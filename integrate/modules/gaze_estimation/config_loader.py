"""
配置文件加載工具
Configuration File Loader

提供讀取和解析 config.yaml 的工具函數
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigLoader:
    """配置文件加載器"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置加載器
        
        Args:
            config_path: 配置文件路徑，如果為 None 則使用默認路徑
        """
        if config_path is None:
            # 默認路徑：項目根目錄下的 config.yaml
            config_path = Path(__file__).parent.parent / 'config.yaml'
        else:
            config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """加載配置文件"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    
    def get(self, *keys, default=None):
        """
        獲取配置值（支持嵌套鍵）
        
        Args:
            *keys: 配置鍵路徑，例如 get('face_detection', 'min_confidence')
            default: 默認值
        
        Returns:
            配置值，如果不存在則返回 default
        
        Examples:
            >>> loader = ConfigLoader()
            >>> loader.get('face_detection', 'min_confidence')
            0.3
            >>> loader.get('model', 'use_gpu')
            True
        """
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
    
    def get_face_detection_config(self) -> Dict[str, Any]:
        """獲取人臉檢測配置"""
        return self.config.get('face_detection', {})
    
    def get_head_pose_config(self) -> Dict[str, Any]:
        """獲取頭部姿態估計配置"""
        return self.config.get('head_pose', {})
    
    def get_normalization_config(self) -> Dict[str, Any]:
        """獲取圖像正規化配置"""
        return self.config.get('normalization', {})
    
    def get_model_config(self) -> Dict[str, Any]:
        """獲取神經網絡模型配置"""
        return self.config.get('model', {})
    
    def get_camera_config(self) -> Dict[str, Any]:
        """獲取相機配置"""
        return self.config.get('camera', {})
    
    def get_output_config(self) -> Dict[str, Any]:
        """獲取輸出配置"""
        return self.config.get('output', {})
    
    def get_debug_config(self) -> Dict[str, Any]:
        """獲取調試配置"""
        return self.config.get('debug', {})
    
    def print_config(self, section: Optional[str] = None):
        """
        打印配置
        
        Args:
            section: 要打印的配置段，如果為 None 則打印全部
        """
        if section:
            config = self.config.get(section, {})
            print(f"\n{'=' * 70}")
            print(f"配置段: {section}")
            print(f"{'=' * 70}")
        else:
            config = self.config
            print(f"\n{'=' * 70}")
            print("完整配置")
            print(f"{'=' * 70}")
        
        print(yaml.dump(config, default_flow_style=False, allow_unicode=True))


def load_config(config_path: Optional[str] = None) -> ConfigLoader:
    """
    快速加載配置文件
    
    Args:
        config_path: 配置文件路徑
    
    Returns:
        ConfigLoader 實例
    
    Examples:
        >>> config = load_config()
        >>> min_confidence = config.get('face_detection', 'min_confidence')
    """
    return ConfigLoader(config_path)


# 單例模式：全局配置對象
_global_config: Optional[ConfigLoader] = None


def get_global_config() -> ConfigLoader:
    """
    獲取全局配置對象（單例模式）
    
    Returns:
        全局 ConfigLoader 實例
    """
    global _global_config
    if _global_config is None:
        _global_config = ConfigLoader()
    return _global_config


if __name__ == '__main__':
    # 測試配置加載
    config = load_config()
    
    print("測試配置加載器")
    print("=" * 70)
    
    # 測試獲取各個配置段
    print("\n1. 人臉檢測配置:")
    print(f"   min_confidence: {config.get('face_detection', 'min_confidence')}")
    print(f"   yolo_conf: {config.get('face_detection', 'yolo_conf')}")
    
    print("\n2. 頭部姿態估計配置:")
    print(f"   face_model_path: {config.get('head_pose', 'face_model_path')}")
    print(f"   use_iterative: {config.get('head_pose', 'use_iterative')}")
    
    print("\n3. 圖像正規化配置:")
    print(f"   focal_length: {config.get('normalization', 'focal_length')}")
    print(f"   distance: {config.get('normalization', 'distance')}")
    print(f"   output_size: {config.get('normalization', 'output_size')}")
    
    print("\n4. 模型配置:")
    print(f"   model_path: {config.get('model', 'model_path')}")
    print(f"   use_gpu: {config.get('model', 'use_gpu')}")
    
    print("\n✓ 配置加載測試完成！")