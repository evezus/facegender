import cv2

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
gender_list = ['male', 'female']
gender_net = cv2.dnn.readNetFromCaffe(
    prototxt="models/deploy_gender.prototxt",
    caffeModel="models/gender_net.caffemodel")


def foto_analyse(img_path):
    image = cv2.imread(img_path, cv2.cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_alt.xml')
    faces = face_cascade.detectMultiScale(image, 1.05, minNeighbors=5, minSize=(10, 10))

    faces_info = []
    for (x, y, w, h) in faces:
        face_img = image[y:y + h, x:x + w].copy()
        blob = cv2.dnn.blobFromImage(face_img, 1.05, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

        # Predict gender
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        r = gender_preds[0][0] * gender_preds[0][1]
        if r <= 0.02:
            gender = gender_list[0]
        else:
            gender = gender_list[1]

        faces_info.append({
            'point': {
                'x': x,
                'y': y,
                'w': w,
                'h': h,
            },
            'gender': gender

        })
        print(gender)
        cv2.imwrite(str(x)+gender+'.jpg', face_img)
        input('n')

    return {'found': len(faces_info), 'faces': faces_info}


if __name__ == '__main__':
    print(foto_analyse('1.jpg'))
