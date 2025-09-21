import torch
import torch.nn.functional as F
from PIL import Image, ImageTk
import numpy as np
import cv2
import argparse
from transformers import AutoImageProcessor, AutoModel
import torchvision.transforms as T
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast
import time
import sys
import os
import glob
import sqlite3
import faiss
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import traceback
from collections import Counter

# 将 samurai 项目添加到 Python 路径
samurai_path = "D:\\\\projects\\\\samurai\\\\sam2"
if samurai_path not in sys.path:
    sys.path.append(samurai_path)

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# --- 模型和图像处理 ---

def load_dinov3_model(model_path='dinov3-vith16plus-pretrain-lvd1689m'):
    """使用 transformers 加载本地 DINOv3 模型"""
    print(f"Loading DINOv3 model from: {model_path}...")
    processor = AutoImageProcessor.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    print(f"DINOv3 model loaded on device: {device}")
    return model, processor, device

def load_sam_model(model_path, config_path, device):
    """加载 SAM 模型"""
    print(f"Loading SAM model from checkpoint: {model_path}...")
    sam_model = build_sam2(config_path, ckpt_path=model_path, device=device)
    sam_model.half()
    predictor = SAM2ImagePredictor(sam_model)
    print("SAM model and predictor loaded.")
    return predictor

def get_feature_map(pil_img, model, processor, device, patch_size=16):
    """
    提取图像的 DINOv3 特征图。
    返回 (C, H, W) 格式的特征图和处理后的图像尺寸。
    """
    w, h = pil_img.size
    new_w = w - (w % patch_size)
    new_h = h - (h % patch_size)
    if new_w == 0 or new_h == 0:
        return None, (0, 0)

    img_resized = pil_img.resize((new_w, new_h), Image.Resampling.BICUBIC)

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=processor.image_mean, std=processor.image_std)
    ])
    img_tensor = transform(img_resized).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        num_register_tokens = 4 # for vith16plus
        start_index = 1 + num_register_tokens
        features = outputs.last_hidden_state[:, start_index:, :]

    h_feat, w_feat = new_h // patch_size, new_w // patch_size
    features_map = features.permute(0, 2, 1).reshape(1, -1, h_feat, w_feat)

    return features_map.squeeze(0), (new_w, new_h)

# --- GUI 应用 ---

class SegmenterApp:
    def __init__(self, master, dinov3_model, dinov3_processor, sam_predictor, device, feature_dim):
        self.master = master
        self.master.title("Interactive Segmentation and Feature Extraction")

        # Models and device
        self.dinov3_model = dinov3_model
        self.dinov3_processor = dinov3_processor
        self.sam_predictor = sam_predictor
        self.device = device

        # Hyperparameters
        self.SIMILARITY_THRESHOLD = 0.6
        self.N_COMPONENTS_PCA = 128
        self.N_CLUSTERS_KMEANS = 5
        self.COLOR_HISTOGRAM_BINS = (8, 8, 8)
        self.COLOR_DISTANCE_THRESHOLD = 0.12

        # Database and Faiss
        self.db_path = 'features.db'
        self.faiss_path = 'features.faiss'
        self.db_conn, self.faiss_index = self.setup_database_and_faiss()

        # UI Setup
        self.setup_ui()

        # State
        self.image_path = None
        self.image_pil = None
        self.image_np = None
        self.features = None
        self.processed_size = None
        self.feature_widgets = {}
        self.last_highlighted_widget = None
        self.load_saved_features()

    def setup_database_and_faiss(self):
        faiss_dim = self.N_COMPONENTS_PCA
        print(f"Initializing Faiss index with dimension after PCA: {faiss_dim}")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        conn.execute("PRAGMA foreign_keys = ON")

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feature_objects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                mask_path TEXT
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feature_vectors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                object_id INTEGER NOT NULL,
                cluster_index INTEGER NOT NULL,
                FOREIGN KEY (object_id) REFERENCES feature_objects (id) ON DELETE CASCADE
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feature_histograms (
                object_id INTEGER PRIMARY KEY,
                histogram BLOB NOT NULL,
                FOREIGN KEY (object_id) REFERENCES feature_objects (id) ON DELETE CASCADE
            )
        ''')
        conn.commit()

        os.makedirs("masks", exist_ok=True)

        if os.path.exists(self.faiss_path):
            print(f"Loading Faiss index from {self.faiss_path}")
            index = faiss.read_index(self.faiss_path)
            if index.d != faiss_dim:
                print(f"Warning: Faiss dimension mismatch. Creating new index.")
                index = faiss.IndexIDMap(faiss.IndexFlatL2(faiss_dim))
        else:
            print("Creating new Faiss index.")
            index = faiss.IndexIDMap(faiss.IndexFlatL2(faiss_dim))
        return conn, index

    def setup_ui(self):
        main_frame = tk.Frame(self.master)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left Frame
        left_frame = tk.Frame(main_frame, bd=2, relief=tk.SUNKEN)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        btn_select_folder = tk.Button(left_frame, text="Select Folder", command=self.select_folder)
        btn_select_folder.pack(pady=5, padx=5, fill=tk.X)
        self.image_listbox = tk.Listbox(left_frame)
        self.image_listbox.pack(fill=tk.Y, expand=True)
        self.image_listbox.bind('<<ListboxSelect>>', self.on_image_select)

        # Middle Frame
        middle_frame = tk.Frame(main_frame, bd=2, relief=tk.SUNKEN)
        middle_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=middle_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect('button_press_event', self.on_click)

        # Right Frame
        right_frame = tk.Frame(main_frame, bd=2, relief=tk.SUNKEN, width=250)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        right_frame.pack_propagate(False)
        feature_label = tk.Label(right_frame, text="Saved Features")
        feature_label.pack(pady=5)
        scroll_canvas = tk.Canvas(right_frame, borderwidth=0, highlightthickness=0)
        scrollbar = tk.Scrollbar(right_frame, orient="vertical", command=scroll_canvas.yview)
        self.scrollable_feature_frame = tk.Frame(scroll_canvas)
        self.scrollable_feature_frame.bind("<Configure>", lambda e: scroll_canvas.configure(scrollregion=scroll_canvas.bbox("all")))
        scroll_canvas.create_window((0, 0), window=self.scrollable_feature_frame, anchor="nw")
        scroll_canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        scroll_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def select_folder(self):
        folder_path = filedialog.askdirectory()
        if not folder_path: return
        self.image_listbox.delete(0, tk.END)
        image_files = glob.glob(os.path.join(folder_path, '*.*[gG]'))
        self.file_map = {os.path.basename(p): p for p in image_files}
        for f in sorted(self.file_map.keys()):
            self.image_listbox.insert(tk.END, f)

    def on_image_select(self, event):
        selected = self.image_listbox.curselection()
        if not selected: return
        filename = self.image_listbox.get(selected[0])
        self.image_path = self.file_map[filename]
        self.load_image()

    def load_image(self):
        if not self.image_path: return
        self.image_pil = Image.open(self.image_path).convert('RGB')
        self.image_np = np.array(self.image_pil)

        print("Setting image for SAM predictor...")
        with autocast(dtype=torch.float16):
            self.sam_predictor.set_image(self.image_np)
        if self.device.type == 'cuda': torch.cuda.synchronize()

        print("Extracting DINOv3 features...")
        self.features, self.processed_size = get_feature_map(self.image_pil, self.dinov3_model, self.dinov3_processor, self.device)

        self.ax.clear()
        self.ax.imshow(self.image_np)
        self.ax.set_title("Click on a point to segment")
        self.canvas.draw()

    def on_click(self, event):
        if event.inaxes != self.ax or self.features is None: return
        x, y = int(event.xdata), int(event.ydata)
        print(f"DEBUG [on_click]: Click event at ({x}, {y})")

        with autocast(dtype=torch.float16):
            masks, _, _ = self.sam_predictor.predict(point_coords=np.array([[x, y]]), point_labels=np.array([1]), multimask_output=True)
        mask = masks[0] # Using the first mask

        try:
            feature_clusters = self._compute_feature_clusters(mask)
            if feature_clusters is None: return
            color_hist = self.compute_color_histogram(mask)
        except Exception as e:
            print(f"Error computing features: {e}"); traceback.print_exc(); return

        found_id, dist = self.search_for_similar_feature(feature_clusters, color_hist)
        print(f"DEBUG: Search -> Closest ID: {found_id}, Dist: {dist:.4f}, Threshold: {self.SIMILARITY_THRESHOLD}")

        if found_id is not None and dist < self.SIMILARITY_THRESHOLD:
            print(f"Found similar feature (ID: {found_id}). Highlighting.")
            self.handle_found_feature(found_id)
        else:
            print("No similar feature found. Adding new one.")
            name = simpledialog.askstring("Input", "Enter feature name:", parent=self.master)
            if name and name.strip():
                self.add_new_feature(name.strip(), feature_clusters, mask)

    def _compute_feature_clusters(self, mask):
        h, w = self.features.shape[1], self.features.shape[2]
        mask_resized = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)

        indices_y, indices_x = np.where(mask_resized)
        if len(indices_y) < self.N_CLUSTERS_KMEANS:
            print(f"Warning: Not enough patches ({len(indices_y)}) for clustering (k={self.N_CLUSTERS_KMEANS}).")
            return None
        print(f"DEBUG [_compute_feature_clusters]: Found {len(indices_y)} patches under the mask.")

        features = self.features[:, indices_y, indices_x].T.cpu().numpy()
        print(f"DEBUG [_compute_feature_clusters]: Raw features shape: {features.shape}")

        n_components = min(self.N_COMPONENTS_PCA, *features.shape)
        if n_components <= 0:
            print(f"DEBUG [_compute_feature_clusters]: Not enough features for PCA ({n_components}).")
            return None
        print(f"DEBUG [_compute_feature_clusters]: Using {n_components} PCA components.")

        pca = PCA(n_components=n_components)
        features_pca = pca.fit_transform(features)
        print(f"DEBUG [_compute_feature_clusters]: Features shape after PCA: {features_pca.shape}")

        kmeans = KMeans(n_clusters=self.N_CLUSTERS_KMEANS, random_state=0, n_init=10)
        cluster_centers = kmeans.fit(features_pca).cluster_centers_

        # L2-normalize the cluster centers before returning
        normalized_centers = normalize(cluster_centers, norm='l2', axis=1)

        print(f"DEBUG [_compute_feature_clusters]: Computed and L2-normalized {self.N_CLUSTERS_KMEANS} cluster centers. Shape: {normalized_centers.shape}")
        return normalized_centers

    def compute_color_histogram(self, mask):
        if self.image_np is None:
            return None

        hsv_image = cv2.cvtColor(self.image_np, cv2.COLOR_RGB2HSV)
        mask_uint8 = (mask.astype(np.uint8) * 255)

        hist = cv2.calcHist(
            [hsv_image],
            [0, 1, 2],
            mask_uint8,
            list(self.COLOR_HISTOGRAM_BINS),
            [0, 180, 0, 256, 0, 256]
        )

        if hist is None or float(hist.sum()) == 0.0:
            bins = int(np.prod(self.COLOR_HISTOGRAM_BINS))
            return np.zeros(bins, dtype=np.float32)

        hist_normalized = cv2.normalize(hist, None, norm_type=cv2.NORM_L1)
        return hist_normalized.flatten().astype(np.float32)

    def _get_color_histogram(self, object_id):
        cursor = self.db_conn.cursor()
        cursor.execute("SELECT histogram FROM feature_histograms WHERE object_id = ?", (object_id,))
        row = cursor.fetchone()
        if not row or row[0] is None:
            return None

        hist = np.frombuffer(row[0], dtype=np.float32)
        expected = int(np.prod(self.COLOR_HISTOGRAM_BINS))
        if hist.size != expected:
            print(f"DEBUG [_get_color_histogram]: Stored histogram size {hist.size} != expected {expected}.")
            return None
        return hist

    def _compare_color_histograms(self, hist1, hist2):
        if hist1 is None or hist2 is None:
            return float('inf')

        hist1_cv = hist1.reshape(-1, 1).astype(np.float32)
        hist2_cv = hist2.reshape(-1, 1).astype(np.float32)
        distance = cv2.compareHist(hist1_cv, hist2_cv, cv2.HISTCMP_BHATTACHARYYA)
        return float(distance)

    def search_for_similar_feature(self, query_clusters, query_color_hist):
        print(f"\n--- DEBUG: Starting New Search ---")
        print(f"DEBUG [search]: Received {query_clusters.shape[0]} query clusters.")
        if self.faiss_index.ntotal == 0:
            print("DEBUG [search]: Faiss index is empty.")
            return None, float('inf')

        print(f"DEBUG [search]: Faiss index contains {self.faiss_index.ntotal} total vectors.")

        cursor = self.db_conn.cursor()
        cursor.execute("SELECT id, object_id FROM feature_vectors")
        db_vectors = cursor.fetchall()
        if not db_vectors:
            print("DEBUG [search]: Database has no vectors.")
            return None, float('inf')


        # L2-normalize the query clusters before searching
        normalized_query_clusters = normalize(query_clusters, norm='l2', axis=1)

        id_map = {v[0]: v[1] for v in db_vectors}
        dists, ids = self.faiss_index.search(normalized_query_clusters.astype(np.float32), 1)

        print(f"DEBUG [search]: Faiss search results (top 1 for each query vector):")
        print(f"DEBUG [search]: Distances:\n{dists.flatten()}")
        print(f"DEBUG [search]: Vector IDs:\n{ids.flatten()}")

        effective_ids = ids.flatten()
        obj_ids = [id_map.get(i) for i in effective_ids if i != -1 and id_map.get(i) is not None]

        if not obj_ids:
            print("DEBUG [search]: No matching object IDs found in the database for the returned vector IDs.")
            return None, float('inf')

        print(f"DEBUG [search]: Mapped vector IDs to object IDs: {obj_ids}")

        id_counts = Counter(obj_ids)
        print(f"DEBUG [search]: Object ID counts: {id_counts}")

        if not id_counts:
            return None, float('inf')

        # --- New Robust Voting Logic ---
        most_common_list = id_counts.most_common(1)
        if not most_common_list:
            print("DEBUG [search]: No common items found.")
            return None, float('inf')

        common_id, count = most_common_list[0]

        # Stricter consensus: more than half of the vectors must agree.
        min_consensus = np.ceil(self.N_CLUSTERS_KMEANS / 2)
        print(f"DEBUG [search]: Most common object ID: {common_id} with count: {count}. Required consensus: >={min_consensus}")

        if count < min_consensus:
            print(f"DEBUG [search]: Consensus not met. Match rejected.")
            return None, float('inf')

        print(f"DEBUG [search]: Consensus met. Proceeding with object ID {common_id}.")

        stored_hist = self._get_color_histogram(common_id)
        if stored_hist is None:
            print(f"DEBUG [search]: No stored color histogram for object ID {common_id}. Rejecting match.")
            return None, float('inf')

        color_distance = self._compare_color_histograms(query_color_hist, stored_hist)
        print(f"DEBUG [search]: Color distance for object ID {common_id}: {color_distance:.4f}. Threshold: {self.COLOR_DISTANCE_THRESHOLD}")

        if color_distance > self.COLOR_DISTANCE_THRESHOLD:
            print(f"DEBUG [search]: Color distance above threshold. Rejecting match.")
            return None, float('inf')

        # --- End of New Logic ---

        relevant_dists = []
        for i, vector_id in enumerate(effective_ids):
            if vector_id != -1 and id_map.get(vector_id) == common_id:
                relevant_dists.append(dists[i][0])

        median_dist = np.median(relevant_dists) if relevant_dists else float('inf')

        print(f"DEBUG [search]: Distances for object ID {common_id}: {relevant_dists}")
        print(f"DEBUG [search]: Final median distance for object ID {common_id}: {median_dist:.4f}")
        print(f"--- DEBUG: Search Finished ---\n")

        return common_id, median_dist

    def handle_found_feature(self, object_id):
        cursor = self.db_conn.cursor()
        cursor.execute("SELECT name, mask_path FROM feature_objects WHERE id = ?", (object_id,))
        res = cursor.fetchone()
        if res:
            name, path = res
            print(f"Visualizing existing feature: '{name}'")
            if os.path.exists(path):
                mask = np.array(Image.open(path))[:, :, 3] > 0
                self.visualize_mask_on_main_canvas(mask)
                self.highlight_feature_in_list(object_id)

    def highlight_feature_in_list(self, object_id):
        if self.last_highlighted_widget and self.last_highlighted_widget.winfo_exists():
            self.last_highlighted_widget.config(bg='SystemButtonFace')
            for child in self.last_highlighted_widget.winfo_children(): child.config(bg='SystemButtonFace')

        target = self.feature_widgets.get(object_id)
        if target and target.winfo_exists():
            target.config(bg="lightblue")
            for child in target.winfo_children(): child.config(bg="lightblue")
            self.last_highlighted_widget = target

            target.update_idletasks()
            canvas = self.scrollable_feature_frame.master
            y = target.winfo_y()
            canvas.yview_moveto(y / self.scrollable_feature_frame.winfo_height())

    def add_new_feature(self, name, clusters, mask):
        try:
            cursor = self.db_conn.cursor()
            cursor.execute("INSERT INTO feature_objects (name) VALUES (?)", (name,))
            obj_id = cursor.lastrowid

            path = os.path.join("masks", f"mask_obj_{obj_id}.png")
            self.save_cutout_image(mask, path)
            cursor.execute("UPDATE feature_objects SET mask_path = ? WHERE id = ?", (path, obj_id))

            color_hist = self.compute_color_histogram(mask)
            if color_hist is None:
                color_hist = np.zeros(int(np.prod(self.COLOR_HISTOGRAM_BINS)), dtype=np.float32)
            cursor.execute("INSERT OR REPLACE INTO feature_histograms (object_id, histogram) VALUES (?, ?)", (obj_id, color_hist.astype(np.float32).tobytes()))

            vec_ids = []
            for i, cluster in enumerate(clusters):
                cursor.execute("INSERT INTO feature_vectors (object_id, cluster_index) VALUES (?, ?)", (obj_id, i))
                vec_ids.append(cursor.lastrowid)
            self.db_conn.commit()

            self.faiss_index.add_with_ids(clusters.astype(np.float32), np.array(vec_ids, dtype=np.int64))
            faiss.write_index(self.faiss_index, self.faiss_path)

            print(f"Saved new feature '{name}' (ID: {obj_id}) with {len(vec_ids)} vectors.")
            self.visualize_mask_on_main_canvas(mask)
            self.add_feature_to_list_display(obj_id, name, path)
        except Exception as e:
            print(f"Error saving: {e}"); self.db_conn.rollback()

    def save_cutout_image(self, mask, path):
        h, w = self.image_np.shape[:2]
        cutout = np.zeros((h, w, 4), dtype=np.uint8)
        b_mask = mask.astype(bool)
        cutout[b_mask, :3] = self.image_np[b_mask]
        cutout[b_mask, 3] = 255
        Image.fromarray(cutout).save(path)

    def visualize_mask_on_main_canvas(self, mask):
        img = self.image_np.copy()
        color = np.random.randint(0, 255, 3)
        img[mask.astype(bool)] = img[mask.astype(bool)] * 0.5 + color * 0.5
        self.ax.clear()
        self.ax.imshow(img)
        self.ax.set_title("Click on a point")
        self.canvas.draw()

    def add_feature_to_list_display(self, obj_id, name, path):
        if obj_id in self.feature_widgets: return

        frame = tk.Frame(self.scrollable_feature_frame, bd=1, relief=tk.SOLID)
        frame.pack(fill=tk.X, pady=2, padx=2)

        photo = None
        if os.path.exists(path):
            img = Image.open(path); img.thumbnail((64, 64)); photo = ImageTk.PhotoImage(img)

        img_label = tk.Label(frame, image=photo); img_label.image = photo; img_label.pack(side=tk.LEFT)
        name_label = tk.Label(frame, text=name, anchor="w", wraplength=150)
        name_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        for w in [frame, img_label, name_label]:
            w.bind("<Double-1>", lambda e, p=path, n=name: self.on_thumbnail_double_click(p, n))
            w.bind("<Button-3>", lambda e, oid=obj_id: self._show_context_menu(e, oid))
        self.feature_widgets[obj_id] = frame

    def load_saved_features(self):
        for w in self.scrollable_feature_frame.winfo_children(): w.destroy()
        self.feature_widgets.clear()
        cursor = self.db_conn.cursor()
        cursor.execute("SELECT id, name, mask_path FROM feature_objects ORDER BY id")
        for obj_id, name, path in cursor.fetchall():
            if path: self.add_feature_to_list_display(obj_id, name, path)

    def _show_context_menu(self, event, obj_id):
        menu = tk.Menu(self.master, tearoff=0)
        menu.add_command(label="Rename", command=lambda: self.rename_feature(obj_id))
        menu.add_command(label="Delete", command=lambda: self.delete_feature(obj_id))
        menu.tk_popup(event.x_root, event.y_root)

    def rename_feature(self, obj_id):
        new_name = simpledialog.askstring("Rename", "New name:", parent=self.master)
        if new_name and new_name.strip():
            cursor = self.db_conn.cursor()
            cursor.execute("UPDATE feature_objects SET name = ? WHERE id = ?", (new_name.strip(), obj_id))
            self.db_conn.commit()
            if obj_id in self.feature_widgets:
                for w in self.feature_widgets[obj_id].winfo_children():
                    if isinstance(w, tk.Label) and not hasattr(w, 'image'):
                        w.config(text=new_name.strip())
                        break

    def delete_feature(self, obj_id):
        if not messagebox.askyesno("Confirm", "Delete this feature permanently?", parent=self.master): return
        try:
            cursor = self.db_conn.cursor()
            cursor.execute("SELECT id FROM feature_vectors WHERE object_id=?", (obj_id,))
            vec_ids = [row[0] for row in cursor.fetchall()]
            cursor.execute("SELECT mask_path FROM feature_objects WHERE id=?", (obj_id,))
            path = cursor.fetchone()[0]

            cursor.execute("DELETE FROM feature_objects WHERE id=?", (obj_id,))
            self.db_conn.commit()

            if vec_ids:
                self.faiss_index.remove_ids(np.array(vec_ids, dtype=np.int64))
                faiss.write_index(self.faiss_index, self.faiss_path)
            if path and os.path.exists(path): os.remove(path)
            if obj_id in self.feature_widgets: self.feature_widgets.pop(obj_id).destroy()
        except Exception as e:
            print(f"Error during deletion: {e}")

    def on_thumbnail_double_click(self, path, name):
        if not os.path.exists(path): return
        viewer = tk.Toplevel(self.master); viewer.overrideredirect(True)
        try:
            img = Image.open(path)
            max_w = int(self.master.winfo_screenwidth() * 0.8)
            max_h = int(self.master.winfo_screenheight() * 0.8)
            img.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            label = tk.Label(viewer, image=photo, bg='white'); label.image = photo; label.pack()
            viewer.update_idletasks()
            x = (self.master.winfo_screenwidth() - viewer.winfo_width()) // 2
            y = (self.master.winfo_screenheight() - viewer.winfo_height()) // 2
            viewer.geometry(f"+{x}+{y}")
            viewer.bind("<Escape>", lambda e: viewer.destroy())
            viewer.bind("<Button-1>", lambda e: viewer.destroy())
        except Exception as e:
            print(f"Error in viewer: {e}")

    def on_close(self):
        if self.db_conn: self.db_conn.close()
        self.master.destroy()

def main():
    parser = argparse.ArgumentParser(description="Interactive Segmentation GUI")
    parser.add_argument('--dinov3_path', type=str, default="dinov3-vith16plus-pretrain-lvd1689m", help="Path to DINOv3 model.")
    parser.add_argument('--sam_ckpt', type=str, default="D:\\\\projects\\\\samurai\\\\sam2\\\\checkpoints\\\\sam2.1_hiera_base_plus.pt", help="Path to SAM checkpoint.")
    parser.add_argument('--sam_config', type=str, default="D:\\\\projects\\\\samurai\\\\sam2\\\\sam2\\\\configs\\\\sam2.1\\\\sam2.1_hiera_b+.yaml", help="Path to SAM config.")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dinov3_model, dinov3_processor, _ = load_dinov3_model(args.dinov3_path)
    sam_predictor = load_sam_model(args.sam_ckpt, args.sam_config, device)

    # Pass the original feature dimension
    feature_dim = dinov3_model.config.hidden_size

    root = tk.Tk()
    app = SegmenterApp(root, dinov3_model, dinov3_processor, sam_predictor, device, feature_dim)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()

if __name__ == '__main__':
    main()
