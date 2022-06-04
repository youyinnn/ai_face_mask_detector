from itertools import count
import cv2
import os
import sys
import dlib
from imutils import face_utils


def crop_boundary(top, bottom, left, right, faces, actual_h, actual_w):
    # if faces:
    #     top = max(0, top - 200)
    #     left = max(0, left - 100)
    #     right += 100
    #     bottom += 100
    # else:
    #     top = max(0, top - 50)
    #     left = max(0, left - 50)
    #     right += 50
    #     bottom += 50

    count = 0
    while top > 0 and left > 0 and bottom < actual_h and right < actual_w and count < 200:
        top -= 1
        left -= 1
        bottom += 1
        right += 1
        count += 1

    return (top, bottom, left, right)


def crop_face(imgpath, dirName, extName, count):
    basename = os.path.basename(imgpath)

    print(f"Start {count}: [{basename}]")
    frame = cv2.imread(imgpath)

    actual_h, actual_w, _ = frame.shape

    basename_without_ext = os.path.splitext(basename)[0]
    if frame is None:
        return print(f"\tInvalid file path: [{imgpath}]")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_detect = dlib.get_frontal_face_detector()
    rects = face_detect(gray, 1)
    if not len(rects):
        print(
            f"\tSorry. HOG could not detect any faces from your image.\n\t[{basename}]")
        fail_crop_img_path = os.path.join(
            dirName, 'fail', f"{basename_without_ext}{extName}")
        no_crop_img = frame[0: actual_h, 0: actual_w]
        cv2.imwrite(fail_crop_img_path, cv2.cvtColor(
            no_crop_img, cv2.COLOR_RGB2BGR))
    else:
        for (i, rect) in enumerate(rects):
            (x, y, w, h) = face_utils.rect_to_bb(rect)

            if (x < 0) or (y < 0):
                # print(f"\tFAIL: [{basename}]")
                fail_crop_img_path = os.path.join(
                    dirName, 'fail', f"{basename_without_ext}{extName}")
                no_crop_img = frame[0: actual_h, 0: actual_w]
                cv2.imwrite(fail_crop_img_path, cv2.cvtColor(
                    no_crop_img, cv2.COLOR_RGB2BGR))
                continue

            top, bottom, left, right = crop_boundary(
                y, y + h, x, x + w, len(rects) <= 2, actual_h, actual_w)
            crop_img_path = os.path.join(
                dirName, 'success', f"{basename_without_ext}_crop_{i}{extName}")

            crop_img = frame[top: bottom, left: right]
            cv2.imwrite(crop_img_path, cv2.cvtColor(
                crop_img, cv2.COLOR_RGB2BGR))

        # return print(f"\tSUCCESS: [{basename}]")


def main(argv):
    extName = ".jpeg"
    dirName = "auto_crop"
    os.makedirs(dirName, exist_ok=True)
    os.makedirs(os.path.join(dirName, 'success'), exist_ok=True)
    os.makedirs(os.path.join(dirName, 'fail'), exist_ok=True)

    if len(argv) == 1:
        sys.exit("Usage: python crop_face.py <image path> ...")
    if len(argv) == 2:
        img_path = os.path.join(os.getcwd(), argv[1])
        lsdir = os.listdir(img_path)
        count = 0
        for img in lsdir:
            count += 1
            crop_face(os.path.join(img_path, img), dirName, extName, count)


if __name__ == "__main__":
    main(sys.argv)
