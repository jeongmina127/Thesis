from yolo_tiler import YoloTiler, TileConfig

src = "/mnt/c/Users/Jeongmin Cho/Desktop/Git Repository/Thesis/WiSARD_Multi_Modal_Sample/210417_MtErie_Enterprise_VIS_0003_split"         # Source YOLO dataset directory
dst = "/mnt/c/Users/Jeongmin Cho/Desktop/Git Repository/Thesis/WiSARD_Multi_Modal_Sample/VIS_tiled"   # Output directory for tiled dataset

config = TileConfig(
    # Size of each tile (width, height). Can be:
    # - Single integer for square tiles: slice_wh=640
    # - Tuple for rectangular tiles: slice_wh=(640, 480)
    slice_wh=(512, 512),

    # Overlap between adjacent tiles. Can be:
    # - Single float (0-1) for uniform overlap percentage: overlap_wh=0.1
    # - Tuple of floats for different overlap in each dimension: overlap_wh=(0.1, 0.1)
    # - Single integer for pixel overlap: overlap_wh=64
    # - Tuple of integers for different pixel overlaps: overlap_wh=(64, 48)
    overlap_wh=(0.1, 0.1),

    # Output image file extension to save (defaults to input extension if None)
    # Set to a specific extension like ".jpg" or ".png" to convert formats
    # Note: Output mask must be .png for semantic_segmentation
    output_ext=None,

    # Type of YOLO annotations to process:
    # - "object_detection": Standard YOLO format (class, x, y, width, height)
    # - "instance_segmentation": YOLO segmentation format (class, x1, y1, x2, y2, ...)
    # - "semantic_segmentation": PNG mask format (0=background, 1-255=class IDs)
    # - "image_classification": YOLO classification format (class)
    annotation_type="object_detection",

    # For instance segmentation only: Controls point density along polygon edges
    # Lower values = more points, higher quality but larger files
    #densify_factor=0.01,

    # For instance segmentation only: Controls polygon smoothing
    # Lower values = more details preserved, higher values = smoother shapes
    #smoothing_tolerance=0.99,

    # Dataset split ratios (must sum to 1.0)
    train_ratio=0.6,  # Proportion of data for training
    valid_ratio=0.2,  # Proportion of data for validation
    test_ratio=0.2,   # Proportion of data for testing

    # Optional margins to exclude from input images. Can be:
    # - Single float (0-1) for uniform margin percentage: margins=0.1
    # - Tuple of floats for different margins: margins=(0.1, 0.1, 0.1, 0.1)
    # - Single integer for pixel margins: margins=64
    # - Tuple of integers for different pixel margins: margins=(64, 64, 64, 64)
    #margins=0.0,

    # Include negative samples (tiles without any instances)
    include_negative_samples=True,

    # Include source data (copied over, and included in the tiled dataset)
    copy_source_data=False,

    # Compression setting (interpreted differently for each format):
    # - JPEG/JPG: Quality level (0-100)
    # - PNG: Automatically converts to compression level (0-9)
    # - TIFF: Selects appropriate compression method based on quality
    #   * High quality (≥90): Lossless LZW compression
    #   * Medium quality (≥75): Lossless DEFLATE compression
    #   * Lower quality (<75): JPEG compression with adjusted quality
    # - BMP: No compression supported
    compression=95
)

tiler = YoloTiler(
    source=src,
    target=dst,
    config=config,
    num_viz_samples=5,                      # Number of samples to visualize
    show_processing_status=True             # Show the progress of the tiling process
)

tiler.run()