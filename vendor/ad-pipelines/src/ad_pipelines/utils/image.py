from typing import Dict, Any, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image
from torchvision.transforms import ToPILImage

to_pil = ToPILImage()

def get_cv2_image_from_pil(pil_image: Image.Image) -> np.ndarray:
    """
    Convert a PIL Image to a cv2-compatible numpy array.

    This function converts a PIL Image object to a numpy array format that is
    compatible with OpenCV (cv2). 

    Args:
        pil_image (Image.Image): A PIL Image object to be converted.
            Supported modes are "RGB" and "L" (grayscale).

    Returns:
        np.ndarray: A numpy array representing the image in cv2-compatible format.
            For RGB images, returns a BGR array. For grayscale images, returns
            a 2D array.
    """
    if pil_image.mode == "RGB":
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    elif pil_image.mode == "L":
        return np.array(pil_image)
    else:
        raise ValueError(f"Unsupported color mode: {pil_image.mode}")

def get_pil_image_from_cv2(cv2_image: np.ndarray) -> Image.Image:
    """
    Convert a CV2/OpenCV image (numpy array) to a PIL Image.
    
    This function handles both grayscale and color images, converting them
    to the appropriate PIL Image format.
    
    Args:
        cv2_image (np.ndarray): Input image as a numpy array from OpenCV.
                               Can be either 2D (grayscale) or 3D (color) array.
    
    Returns:
        Image.Image: PIL Image object with appropriate mode ('L' for grayscale,
                    'RGB' for color images).
    """
    if cv2_image.ndim == 2:  # Grayscale image
        return Image.fromarray(cv2_image, mode='L')
    elif cv2_image.shape[2] == 3:  # Color image
        return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))
    else:
        raise ValueError(f"Unsupported image shape: {cv2_image.shape}")
    
def ensure_pil_image(image: Union[Image.Image, np.ndarray]) -> Image.Image:
    """
    Ensure the input is a PIL Image. If it's a numpy array, convert it.
    Args:
        image (Union[Image.Image, np.ndarray]): Input image, either as a PIL Image
            or a numpy array.
    Returns:
        Image.Image: The input image as a PIL Image.
    """
    if isinstance(image, Image.Image):
        return image
    elif isinstance(image, np.ndarray):
        return get_pil_image_from_cv2(image)
    else:
        raise ValueError("Input must be a PIL Image or a numpy array.")
    
def get_heatmap(image: Union[Image.Image, np.ndarray], color_map: int = cv2.COLORMAP_JET) -> Union[Image.Image, np.ndarray]:
    """
    Convert an image to a heatmap representation.

    Args:
        image (Union[Image.Image, np.ndarray]): A grayscale image representation.
        color_map (int): OpenCV colormap to apply. Defaults to cv2.COLORMAP_JET.

    Returns:
        Union[Image.Image, np.ndarray]: The image converted to the heatmap.
    """
    is_pil = isinstance(image, Image.Image)
    if is_pil:
        image = get_cv2_image_from_pil(image)

    heatmap = cv2.applyColorMap(image, color_map)

    if is_pil:
        heatmap = get_pil_image_from_cv2(heatmap)

    return heatmap

def get_heatmap_overlay(
    src_image: Union[Image.Image, np.ndarray], 
    image: Optional[Union[Image.Image, np.ndarray]] = None,
    heatmap: Optional[Union[Image.Image, np.ndarray]] = None,
    alpha: float = 0.5
) -> Union[Image.Image, np.ndarray]:
    """
    Overlay a heatmap on top of an image.

    Args:
        src_image (Union[Image.Image, np.ndarray]): The source image to overlay the heatmap on.
        image (Optional[Union[Image.Image, np.ndarray]]): The image to be converted to heatmap.
            Either image or heatmap must be provided.
        heatmap (Optional[Union[Image.Image, np.ndarray]]): The pre-converted heatmap to overlay.
            Either image or heatmap must be provided.
        alpha (float): The transparency level of the overlay. Defaults to 0.5.

    Returns:
        Union[Image.Image, np.ndarray]: The source image with the heatmap overlay.

    Raises:
        ValueError: If both image and heatmap are None.
    """
    if image is None and heatmap is None:
        raise ValueError("Either 'image' or 'heatmap' must be provided.")
    
    is_pil = isinstance(src_image, Image.Image)
    
    if is_pil:
        src_image = get_cv2_image_from_pil(src_image)

    # Ensure src_image is in BGR format for overlay
    if src_image.ndim == 2:  # Grayscale
        src_image = cv2.cvtColor(src_image, cv2.COLOR_GRAY2BGR)

    if heatmap is None:
        if isinstance(image, Image.Image):
            image = get_cv2_image_from_pil(image)
        heatmap = get_heatmap(image)
    elif isinstance(heatmap, Image.Image):
        heatmap = get_cv2_image_from_pil(heatmap)

    overlay = cv2.addWeighted(src_image, 1 - alpha, heatmap, alpha, 0)

    if is_pil:
        overlay = get_pil_image_from_cv2(overlay)

    return overlay

def concat_images(images: list[Image.Image]) -> Image.Image:
    """
    Concatenate a list of PIL images horizontally.
    
    Args:
        images (list[Image.Image]): List of PIL Image objects to concatenate.
                                    All images must have the same height.
    
    Returns:
        Image.Image: A new PIL Image with all input images concatenated horizontally.
        
    Raises:
        ValueError: If the list is empty or if images have different heights.
    """
    if not images:
        raise ValueError("Image list cannot be empty.")
    
    if len(images) == 1:
        return images[0].copy()
    
    # Check that all images have the same height
    first_height = images[0].height
    for i, img in enumerate(images):
        if img.height != first_height:
            raise ValueError(f"All images must have the same height. Image {i} has height {img.height}, expected {first_height}.")
    
    # Calculate total width
    total_width = sum(img.width for img in images)
    
    # Create new image with combined width
    result = Image.new(images[0].mode, (total_width, first_height))
    
    # Paste images horizontally
    x_offset = 0
    for img in images:
        result.paste(img.convert("RGB"), (x_offset, 0))
        x_offset += img.width
    
    return result

def rotate_image(image: Union[Image.Image, np.ndarray], angle: float) -> Union[Image.Image, np.ndarray]:
    """
    Rotate an image by a specified angle.

    Args:
        image (Union[Image.Image, np.ndarray]): The input image to be rotated.
        angle (float): The rotation angle in degrees. Positive values mean
                       counter-clockwise rotation.

    Returns:
        Union[Image.Image, np.ndarray]: The rotated image.
    """
    is_pil = isinstance(image, Image.Image)
    if is_pil:
        image = get_cv2_image_from_pil(image)

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # Compute the rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    if is_pil:
        rotated = get_pil_image_from_cv2(rotated)

    return rotated