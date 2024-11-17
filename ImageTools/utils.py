import cv2

def calculate_padding(image):
  """
  Expects image of shape (h, w, c)
  """
  height, width = image.shape[:-1]
  if width == height:
      return 0, 0, 0, 0
  elif width > height:
      delta = 0
      if (width - height)%2 != 0:
        delta = 1
      padding = (0, (width - height) // 2, 0, (width - height) // 2 + delta)

  else:
      delta = 0
      if (height - width)%2 != 0:
        delta = 1
      padding = ((height - width) // 2, 0, (height - width) // 2 + delta, 0)
  return padding

def edge_aware_pad(image, padding):
    height, width = image.shape[:2]
    top = padding[3]
    bottom = padding[1]
    left = padding[2]
    right = padding[0]
    padded_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_REPLICATE)
    return padded_image