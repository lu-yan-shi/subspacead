import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import SpectralClustering, KMeans, HDBSCAN
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score

class AutoSphericalKMeans:
    """
    Spherical KMeans with automatic K selection via Silhouette Score.
    
    Uses L2-normalized features so Euclidean distance equals cosine distance.
    Searches K in [min_k, max_k] and selects the one with highest silhouette score.
    """
    
    def __init__(self, min_k=2, max_k=6):
        self.min_k = min_k
        self.max_k = max_k
        self.best_k = None
        self.labels_ = None
        self.cluster_centers_ = None
        
    def fit_predict(self, X):
        """
        Fit and predict cluster labels.
        
        Args:
            X: (N, D) feature matrix.
            
        Returns:
            labels: (N,) cluster assignments.
        """
        # L2 normalize (makes Euclidean distance equivalent to Cosine distance)
        X_norm = normalize(X, norm='l2', axis=1)
        
        best_score = -1.0
        best_labels = None
        best_k = self.min_k
        
        scores = []
        
        print(f"Searching optimal K from {self.min_k} to {self.max_k}...")
        
        # Search over K values
        for k in range(self.min_k, self.max_k + 1):
            # Run standard KMeans
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_norm)
            
            # Evaluate with Silhouette Score (cosine metric for spherical clustering)
            score = silhouette_score(X_norm, labels, metric='cosine')
            scores.append(score)
            
            print(f"  K={k}: Silhouette Score = {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_labels = labels
                best_k = k
                self.cluster_centers_ = kmeans.cluster_centers_
        
        self.best_k = best_k
        self.labels_ = best_labels
        print(f"-> Selected Optimal K = {self.best_k} (Score: {best_score:.4f})")
        
        return best_labels

class ForegroundAwareSpectralClustering(BaseEstimator, ClusterMixin):
    """
    Adaptive Spectral Clustering with automatic Object/Texture mode detection.
    
    Key Features:
        - Auto mode detection via Coefficient of Variation (CV) of saliency
        - Object mode: Otsu-based fg/bg separation, then cluster foreground only
        - Texture mode: Cluster entire image (no fg/bg separation)
        - Auto K selection via Eigengap heuristic
    
    Theory:
        - CV = std/mean of CLS-Patch similarity measures saliency dispersion
        - Object images: high CV (distinct fg/bg), Texture images: low CV (uniform)
        - Eigengap: largest gap in Laplacian eigenvalues indicates optimal K
    
    Attributes:
        labels_: Cluster labels (0=background for object mode, 1..K=clusters)
        saliency_: CLS-Patch cosine similarity map
        fg_mask_: Boolean foreground mask
        mode_: Detected mode ('object' or 'texture')
        cv_: Coefficient of variation
        auto_k_: Auto-determined K value
    """
    
    # CV threshold: below this is texture (concentrated distribution)
    CV_THRESHOLD = 0.15
    
    def __init__(self, n_clusters='auto', spatial_weight=0.05, max_k=10):
        """
        Args:
            n_clusters: 'auto' (Eigengap heuristic) or int (fixed K).
            spatial_weight: Weight for spatial coords. Lower=feature-driven,
                           higher=spatially contiguous. Default: 0.05.
            max_k: Maximum K for auto mode. Default: 8.
        """
        self.n_clusters = n_clusters
        self.spatial_weight = spatial_weight
        self.max_k = max_k
        self.labels_ = None
        self.saliency_ = None
        self.fg_mask_ = None
        self.mode_ = None  # 'object' or 'texture'
        self.cv_ = None    # Coefficient of Variation
        self.auto_k_ = None  # Auto-determined K (when n_clusters='auto')
    
    def _detect_mode(self, saliency):
        """
        Detect image type using Coefficient of Variation (CV = std/mean).
        
        CV is a dimensionless measure of dispersion:
            - Texture: uniform saliency -> low CV
            - Object: distinct fg/bg -> high CV
        
        Returns:
            str: 'texture' or 'object'
        """
        mean_s = np.mean(saliency)
        std_s = np.std(saliency)
        
        # Avoid division by zero
        cv = std_s / (mean_s + 1e-10)
        self.cv_ = cv
        
        if cv < self.CV_THRESHOLD:
            self.mode_ = 'texture'
        else:
            self.mode_ = 'object'
        
        return self.mode_
    
    def _otsu_threshold(self, values):
        """
        Compute Otsu's optimal threshold via inter-class variance maximization.
        
        Otsu maximizes sigma_b^2 = w0*w1*(mu0-mu1)^2, which is statistically
        optimal for bimodal distributions without any hyperparameters.
        
        Returns:
            float: Optimal threshold value.
        """
        val_min, val_max = values.min(), values.max()
        if val_max - val_min < 1e-6:
            return val_min  # Degenerate case
        
        nbins = 256
        hist, bin_edges = np.histogram(values, bins=nbins, range=(val_min, val_max))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        hist_norm = hist.astype(np.float64) / hist.sum()
        
        # Cumulative sums for efficient computation
        omega = np.cumsum(hist_norm)  # ω(t)
        mu = np.cumsum(hist_norm * bin_centers)  # μ(t) * ω(t)
        mu_total = mu[-1]
        
        # Avoid division by zero
        omega = np.clip(omega, 1e-10, 1.0 - 1e-10)
        
        # Inter-class variance: σ²_B = [μ_T·ω(t) - μ(t)]² / [ω(t)·(1-ω(t))]
        sigma_b_sq = (mu_total * omega - mu) ** 2 / (omega * (1 - omega))
        
        optimal_idx = np.argmax(sigma_b_sq)
        return bin_centers[optimal_idx]
    
    def _compute_saliency(self, patch_tokens, cls_token):
        """
        Compute normalized saliency map via CLS-Patch cosine similarity.
        
        Args:
            patch_tokens: (N, D) patch features.
            cls_token: (D,) CLS token.
            
        Returns:
            saliency_norm: (N,) saliency in [0, 1], max-normalized.
            patch_norm: (N, D) L2-normalized patch features.
        """
        # L2 normalize
        patch_norm = patch_tokens / (np.linalg.norm(patch_tokens, axis=1, keepdims=True) + 1e-10)
        cls_norm = cls_token / (np.linalg.norm(cls_token) + 1e-10)
        
        # Cosine similarity
        saliency = np.dot(patch_norm, cls_norm)
        
        # Normalize to [0, 1] by max (preserves relative ordering)
        saliency_norm = saliency / (saliency.max() + 1e-10)
        
        return saliency_norm, patch_norm
    
    def _build_spatial_features(self, features, fg_indices, H, W):
        """
        Augment features with normalized spatial coordinates.
        
        Adding spatial coords encourages spatially contiguous segmentation
        when feature similarity is ambiguous.
        
        Args:
            features: (N_fg, D) normalized patch features.
            fg_indices: Indices of foreground patches.
            H, W: Spatial dimensions.
            
        Returns:
            (N_fg, D+2) augmented and re-normalized features.
        """
        # Full coordinate grid
        y, x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        coords_full = np.stack([y.flatten(), x.flatten()], axis=1).astype(np.float64)
        coords_full = coords_full / max(H, W)  # Normalize to [0, 1]
        
        # Extract foreground coordinates
        coords_fg = coords_full[fg_indices]
        
        # Concatenate and re-normalize
        augmented = np.concatenate([features, self.spatial_weight * coords_fg], axis=1)
        augmented = augmented / (np.linalg.norm(augmented, axis=1, keepdims=True) + 1e-10)
        
        return augmented
    
    def _eigengap_k(self, affinity, max_k=None):
        """
        Determine optimal K using the Eigengap heuristic.
        
        Theory (Spectral Graph Theory):
            - Laplacian eigenvalues: lambda_1 <= lambda_2 <= ... <= lambda_n
            - For K connected components, first K eigenvalues ~ 0
            - Eigengap = lambda_{k+1} - lambda_k; largest gap indicates optimal K
        
        Args:
            affinity: (N, N) affinity matrix.
            max_k: Maximum K to consider.
            
        Returns:
            int: Optimal K with largest eigengap.
        """
        if max_k is None:
            max_k = self.max_k
        
        n = affinity.shape[0]
        max_k = min(max_k, n - 1)
        
        if max_k < 2:
            return 2
        
        # Compute normalized Laplacian: L_sym = I - D^(-1/2) A D^(-1/2)
        degrees = np.sum(affinity, axis=1)
        degrees = np.clip(degrees, 1e-10, None)
        d_inv_sqrt = 1.0 / np.sqrt(degrees)
        
        norm_affinity = affinity * np.outer(d_inv_sqrt, d_inv_sqrt)
        laplacian = np.eye(n) - norm_affinity
        
        try:
            eigenvalues = np.linalg.eigvalsh(laplacian)
        except np.linalg.LinAlgError:
            return min(5, max_k)
        
        eigenvalues = np.sort(eigenvalues)
        
        # gaps[i] = lambda_{i+1} - lambda_i, corresponds to K = i+1
        gaps = np.diff(eigenvalues[:max_k + 1])
        
        if len(gaps) < 2:
            return 2
        
        # Select K with largest eigengap
        optimal_idx = np.argmax(gaps)
        optimal_k = max(2, optimal_idx + 1)
        
        self.auto_k_ = optimal_k
        return optimal_k
    
    def fit_predict(self, patch_tokens, cls_token, image_shape=None):
        """
        Args:
            patch_tokens: (N, D) DINO patch features.
            cls_token: (D,) DINO CLS token.
            image_shape: Optional (H, W). Infers from N if None.
            
        Returns:
            labels: (N,) array where:
                - Object mode: 0=background, 1..K=foreground clusters
                - Texture mode: 1..K=clusters (no background)
        """
        N, D = patch_tokens.shape
        
        # 1. Infer spatial shape
        if image_shape is None:
            side = int(np.sqrt(N))
            if side * side != N:
                raise ValueError(f"N={N} is not a perfect square. Please provide 'image_shape'.")
            H, W = side, side
        else:
            H, W = image_shape
        
        # 2. Compute saliency map
        saliency, patch_norm = self._compute_saliency(patch_tokens, cls_token)
        self.saliency_ = saliency
        
        # 3. Detect mode: Object vs Texture
        mode = self._detect_mode(saliency)
        
        if mode == 'texture':
            return self._cluster_texture(patch_norm, N, H, W)
        else:
            return self._cluster_object(patch_norm, saliency, N, H, W)
    
    def _cluster_texture(self, patch_norm, N, H, W):
        """
        Texture mode: cluster entire image without fg/bg separation.
        
        For texture images, all patches are semantically similar,
        so we cluster the whole image directly.
        
        Returns:
            labels: (N,) cluster labels starting from 1.
        """
        # All patches are foreground
        self.fg_mask_ = np.ones(N, dtype=bool)
        fg_indices = np.arange(N)
        
        # Spatial augmentation
        fg_augmented = self._build_spatial_features(patch_norm, fg_indices, H, W)
        
        # Build affinity matrix
        affinity = np.dot(fg_augmented, fg_augmented.T)
        affinity = np.clip(affinity, 0, 1)
        np.fill_diagonal(affinity, 0)
        
        # Determine K
        if self.n_clusters == 'auto':
            effective_k = self._eigengap_k(affinity, max_k=min(self.max_k, N // 20))
        else:
            effective_k = min(self.n_clusters, max(2, N // 20))
        
        sc = SpectralClustering(
            n_clusters=effective_k,
            affinity='precomputed',
            random_state=42,
            n_init=10,
            assign_labels='kmeans'
        )
        labels = sc.fit_predict(affinity)
        
        # Shift labels to start from 1 (for consistency with object mode)
        labels = labels + 1
        
        self.labels_ = labels
        return labels
    
    def _cluster_object(self, patch_norm, saliency, N, H, W):
        """
        Object mode: separate fg/bg via Otsu, then cluster foreground only.
        
        For object images, Otsu threshold separates foreground from background,
        then spectral clustering is applied only to foreground patches.
        
        Returns:
            labels: (N,) where 0=background, 1..K=foreground clusters.
        """
        # Otsu threshold for foreground/background separation
        otsu_thresh = self._otsu_threshold(saliency)
        fg_mask = saliency >= otsu_thresh
        self.fg_mask_ = fg_mask
        
        fg_indices = np.where(fg_mask)[0]
        n_fg = len(fg_indices)
        
        # Initialize labels (background = 0)
        labels = np.zeros(N, dtype=np.int32)
        
        # Edge case: too few foreground patches
        min_patches = 2 if self.n_clusters == 'auto' else self.n_clusters
        if n_fg < min_patches:
            if n_fg > 0:
                labels[fg_indices] = 1
            self.labels_ = labels
            return labels
        
        # Extract foreground features
        fg_features = patch_norm[fg_indices]
        
        # Spatial augmentation for foreground only
        fg_augmented = self._build_spatial_features(fg_features, fg_indices, H, W)
        
        # Build foreground-only affinity matrix
        affinity_fg = np.dot(fg_augmented, fg_augmented.T)
        affinity_fg = np.clip(affinity_fg, 0, 1)
        np.fill_diagonal(affinity_fg, 0)
        
        # Determine K
        if self.n_clusters == 'auto':
            effective_k = self._eigengap_k(affinity_fg, max_k=min(self.max_k, n_fg // 10))
        else:
            effective_k = min(self.n_clusters, max(2, n_fg // 10))
        
        sc = SpectralClustering(
            n_clusters=effective_k,
            affinity='precomputed',
            random_state=42,
            n_init=10,
            assign_labels='kmeans'
        )
        fg_labels = sc.fit_predict(affinity_fg)
        
        # Map foreground labels back (shift by 1 so background=0)
        labels[fg_indices] = fg_labels + 1
        
        self.labels_ = labels
        return labels
    
    def get_diagnostics(self):
        """
        Return diagnostic information for debugging.
        
        Returns:
            dict: Contains saliency, fg_mask, labels, mode, cv, auto_k, n_clusters.
        """
        return {
            'saliency': self.saliency_,
            'fg_mask': self.fg_mask_,
            'labels': self.labels_,
            'mode': self.mode_,
            'cv': self.cv_,
            'auto_k': self.auto_k_,
            'n_clusters': len(np.unique(self.labels_)) - (1 if self.mode_ == 'object' else 0)
        }
    
class ForegroundAwareKMeans(ForegroundAwareSpectralClustering):
    """
    Foreground-Aware KMeans with automatic Object/Texture mode detection.
    
    Inherits from AutoSphericalKMeans and adds fg/bg separation for object images.
    Uses Coefficient of Variation (CV) of saliency to detect image type.
    """
    
    def _cluster_object(self, patch_norm, saliency, N, H, W):
        """
        Object mode: separate fg/bg via Otsu, then cluster foreground only.
        
        For object images, Otsu threshold separates foreground from background,
        then spectral clustering is applied only to foreground patches.
        
        Returns:
            labels: (N,) where 0=background, 1..K=foreground clusters.
        """
        # Otsu threshold for foreground/background separation
        otsu_thresh = self._otsu_threshold(saliency)
        fg_mask = saliency >= otsu_thresh
        self.fg_mask_ = fg_mask
        
        fg_indices = np.where(fg_mask)[0]
        n_fg = len(fg_indices)
        
        # Initialize labels (background = 0)
        labels = np.zeros(N, dtype=np.int32)
        
        # Edge case: too few foreground patches
        min_patches = 2 if self.n_clusters == 'auto' else self.n_clusters
        if n_fg < min_patches:
            if n_fg > 0:
                labels[fg_indices] = 1
            self.labels_ = labels
            return labels
        
        # Extract foreground features
        fg_features = patch_norm[fg_indices]
        
        # Spatial augmentation for foreground only
        fg_augmented = self._build_spatial_features(fg_features, fg_indices, H, W)
        
        # Build foreground-only affinity matrix
        affinity_fg = np.dot(fg_augmented, fg_augmented.T)
        affinity_fg = np.clip(affinity_fg, 0, 1)
        np.fill_diagonal(affinity_fg, 0)
        
        # Determine K
        if self.n_clusters == 'auto':
            effective_k = self._eigengap_k(affinity_fg, max_k=min(self.max_k, n_fg // 10))
        else:
            effective_k = min(self.n_clusters, max(2, n_fg // 10))
        
        sc = KMeans(
            n_clusters=effective_k,
            affinity='precomputed',
            random_state=42,
            n_init=10,
            assign_labels='kmeans'
        )
        fg_labels = sc.fit_predict(affinity_fg)
        
        # Map foreground labels back (shift by 1 so background=0)
        labels[fg_indices] = fg_labels + 1
        
        self.labels_ = labels
        return labels
    
class ForegroundAwareHDBSCAN(ForegroundAwareSpectralClustering):
    """
    Adaptive HDBSCAN Clustering with automatic Object/Texture mode detection.
    
    Inherits logic for Saliency calculation, Otsu thresholding, and Mode detection 
    from ForegroundAwareSpectralClustering, but replaces the core clustering 
    engine with HDBSCAN.
    
    Changes:
        - No longer computes full NxN affinity matrix (Memory efficient).
        - 'n_clusters' is ignored; clustering is density-driven.
        - HDBSCAN Noise (-1) is mapped to 0 (Background).
    """

    def __init__(self, n_clusters=5, min_samples=None, spatial_weight=0.05, metric='euclidean'):
        """
        Args:
            min_cluster_size (int): The minimum size of clusters.
            min_samples (int): The number of samples in a neighbourhood for a point to be considered a core point.
            spatial_weight (float): Weight for spatial coords.
            metric (str): Distance metric for HDBSCAN.
        """
        # Initialize parent with dummy n_clusters (not used)
        super().__init__(n_clusters=n_clusters, spatial_weight=spatial_weight)
        
        self.min_cluster_size = n_clusters
        self.min_samples = min_samples
        self.metric = metric

    def _run_hdbscan(self, features):
        """
        Helper to run HDBSCAN and remap labels.
        
        Mapping logic:
            HDBSCAN -1 (Noise) -> 0 (Background)
            HDBSCAN 0, 1, ...  -> 1, 2, ... (Foreground Clusters)
        """
        hdb = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric=self.metric
        )
        labels = hdb.fit_predict(features)
        
        # Remap: -1 becomes 0, 0 becomes 1, etc.
        # This aligns with the convention: 0 is background/noise
        mapped_labels = labels + 1
        
        return mapped_labels

    def _cluster_texture(self, patch_norm, N, H, W):
        """
        Texture mode: Run HDBSCAN on entire image features.
        Overrides parent method to skip affinity matrix computation.
        """
        # All patches are foreground
        self.fg_mask_ = np.ones(N, dtype=bool)
        fg_indices = np.arange(N)
        
        # Spatial augmentation (Reusing parent method)
        fg_augmented = self._build_spatial_features(patch_norm, fg_indices, H, W)
        
        # Run HDBSCAN directly on features (O(N log N)) instead of Affinity (O(N^2))
        labels = self._run_hdbscan(fg_augmented)
        
        self.labels_ = labels
        # Update n_clusters based on detected clusters (excluding background 0)
        self.auto_k_ = len(set(labels)) - (1 if 0 in labels else 0)
        
        return labels

    def _cluster_object(self, patch_norm, saliency, N, H, W):
        """
        Object mode: Separate fg/bg via Otsu (Parent logic), then run HDBSCAN on foreground.
        Overrides parent method.
        """
        # 1. Otsu thresholding (Reusing parent method)
        otsu_thresh = self._otsu_threshold(saliency)
        fg_mask = saliency >= otsu_thresh
        self.fg_mask_ = fg_mask
        
        fg_indices = np.where(fg_mask)[0]
        n_fg = len(fg_indices)
        
        # Initialize labels (background = 0)
        labels = np.zeros(N, dtype=np.int32)
        
        # Edge case: too few foreground patches
        if n_fg < self.min_cluster_size:
            if n_fg > 0:
                # Treat all as a single noise/bg block or single cluster
                # Here we map them to cluster 1 to indicate detected foreground
                labels[fg_indices] = 1 
            self.labels_ = labels
            self.auto_k_ = 1 if n_fg > 0 else 0
            return labels
        
        # 2. Extract and Augment Foreground Features
        fg_features = patch_norm[fg_indices]
        fg_augmented = self._build_spatial_features(fg_features, fg_indices, H, W)
        
        # 3. Run HDBSCAN on foreground only
        fg_labels_mapped = self._run_hdbscan(fg_augmented)
        
        # 4. Map back to full image
        # Note: _run_hdbscan already mapped -1->0, 0->1.
        # However, 0 from HDBSCAN (mapped to 1) should stay 1.
        # Noise from HDBSCAN (mapped to 0) merges with Otsu background (0).
        labels[fg_indices] = fg_labels_mapped
        
        self.labels_ = labels
        self.auto_k_ = len(set(labels)) - 1 # Subtract background 0
        return labels
    
    # _eigengap_k is unused in HDBSCAN version, effectively deprecated.


# clusterer = AutoSpectralClustering(max_k=6, min_k=2)
# labels = clusterer.fit_predict(features)
# mask = labels.reshape(32, 32)