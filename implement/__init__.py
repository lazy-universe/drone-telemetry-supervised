# Package initialization for implement
import torch

RAPIDS_ACTIVE = False
if torch.cuda.is_available():
    try:
        import cuml.accel
        cuml.accel.install()
        RAPIDS_ACTIVE = True
        print("[RAPIDS] cuML instant GPU acceleration enabled successfully.")
        
        # Monkeypatch cuml.accel.estimator_proxy.ProxyBase.__delattr__ to prevent
        # AttributeError crashes during GridSearchCV / hyperparameter tuning due to sklearn 1.9+ callbacks
        import cuml.accel.estimator_proxy as ep
        original_delattr = ep.ProxyBase.__delattr__
        
        def custom_delattr(self, name: str) -> None:
            try:
                original_delattr(self, name)
            except AttributeError:
                if name in self.__dict__:
                    del self.__dict__[name]
                else:
                    pass
                    
        ep.ProxyBase.__delattr__ = custom_delattr
        print("[RAPIDS] Applied scikit-learn 1.9+ compatibility monkeypatch to cuML ProxyBase.")
    except ImportError:
        print("[RAPIDS] cuML instant GPU acceleration not installed. Run 'pip install cuml-cu12 --extra-index-url=https://pypi.nvidia.com' to enable.")

