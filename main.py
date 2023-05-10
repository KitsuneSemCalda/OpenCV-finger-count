import sys
from typing import List

import cv2
import mediapipe as mp


video = cv2.VideoCapture(0)
mpDraw = mp.solutions.drawing_utils
hand = mp.solutions.hands
Hand = hand.Hands(max_num_hands=1)


def main():
    while True:
        check, img = video.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = Hand.process(imgRGB)

        handPoints = results.multi_hand_landmarks

        h, w, _ = img.shape
        pontos = []

        if handPoints:
            for points in handPoints:
                mpDraw.draw_landmarks(img, points, hand.HAND_CONNECTIONS)
                for id,cords in enumerate(points.landmark):
                    cx, cy = int(cords.x * w), int(cords.y * h)
                    cv2.putText(img, str(id),(cx, cy+20), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255, 255, 255), 2)
                    pontos.append((cx,cy))

                dedos = [8, 12, 16, 20]
                contador = 0

                if points:
                    if pontos[4][0] < pontos[2][0]:
                        contador += 1
                    for x in dedos:
                        if pontos[x][1] < pontos[x-2][1]:
                            contador += 1
                cv2.rectangle(img, (80, 10), (200, 100), (255,0,0), -1)
                cv2.putText(img, str(contador), (100, 100), cv2.FONT_HERSHEY_SIMPLEX,4,(255,255,255),5)

        cv2.imshow("Webcam", img)
        cv2.waitKey(1)
    pass


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)