import cv2

mser = cv2.MSER_create()

Im_Text = cv2.imread('text image (3).jpg');
Im_Gray = cv2.cvtColor(Im_Text, cv2.COLOR_BGR2GRAY);
Im_Clone = Im_Text.copy()

regions, _ = mser.detectRegions(Im_Gray)
hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
cv2.polylines(Im_Clone, hulls, 1, (0, 255, 0))

cv2.imshow('mser', Im_Clone)

cv2.waitKey(0)
cv2.destroyAllWindows()