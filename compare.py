# Python program for face
# comparison


from __future__ import print_function, unicode_literals
from facepplib import FacePP, exceptions
import emoji


# define global variables
face_detection = ""
faceset_initialize = ""
face_search = ""
face_landmarks = ""
dense_facial_landmarks = ""
face_attributes = ""
beauty_score_and_emotion_recognition = ""

# define face comparing function
def face_comparing(app, Image1, Image2, im_name):
	
	print()
	print('-'*30)
	print('Comparing Photographs......')
	print('-'*30)


	cmp_ = app.compare.get(image_url1 = Image1,
						image_url2 = Image2)

	print('Photo1', '=', cmp_.image1)
	print('Photo2', '=', cmp_.image2)

	print("The "+im_name +" implimentation has accuracy of: " +str(cmp_.confidence)+ "%")

		
# Driver Code
if __name__ == '__main__':

	# api details
	api_key ='xQLsTmMyqp1L2MIt7M3l0h-cQiy0Dwhl'
	api_secret ='TyBSGw8NBEP9Tbhv_JbQM18mIlorY6-D'

	try:
		
		# call api
		app_ = FacePP(api_key = api_key,
					api_secret = api_secret)
		funcs = [
			face_detection,
			faceset_initialize,
			face_search,
			face_landmarks,
			dense_facial_landmarks,
			face_attributes,
			beauty_score_and_emotion_recognition
		]
		
		# Pair 1
		image1 = 'https://i.postimg.cc/vD51K85H/Screen-Shot-2021-11-17-at-7-58-16-PM.png'
		image2 = 'https://i.postimg.cc/8jDFg8bd/Screen-Shot-2021-11-17-at-7-58-39-PM.png'
		face_comparing(app_, image1, image2, "New")
		
		# Pair2
		image1 = 'https://i.postimg.cc/8jDFg8bd/Screen-Shot-2021-11-17-at-7-58-39-PM.png'
		image2 = 'https://i.postimg.cc/GHQ4cMMW/Screen-Shot-2021-11-17-at-7-58-28-PM.png'
		face_comparing(app_, image1, image2, "Original")		

	except exceptions.BaseFacePPError as e:
		print('Error:', e)
