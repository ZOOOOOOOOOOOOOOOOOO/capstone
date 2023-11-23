from calculate_angle import calculate_angle

def address(first_ankle_center_x,landmarks_dict,Time,current_time,image_width,address_tmp):
    right_wrist = (landmarks_dict['left_wrist'][0] *image_width)
    if(address_tmp==0 and (first_ankle_center_x > right_wrist) ): #오차값 5부여 (x좌표가 동일하지 않는 경우가 존재
        address_tmp = address_tmp + 1
        Time['address'] = current_time
    return Time['address'],address_tmp


def backswing(first_right_shoulder_y,landmarks_dict,Time,current_time,image_height,back_tmp):
    left_pinky = (landmarks_dict['left_pinky'][1] *image_height)
    if(back_tmp==0 and (first_right_shoulder_y > left_pinky) ): #오차값 5부여 (x좌표가 동일하지 않는 경우가 존재
        back_tmp = back_tmp + 1
        Time['back'] = current_time
        print('back',Time['back'])
    return Time['back'],back_tmp

#현재 프레임,다음 프레임 
#def top()

#def impact(first_ankle_center_x,landmarks_dict,Time,current_time,image_width,impact_tmp):
#    right_wrist = (landmarks_dict['left_wrist'][0] *image_width)
#    if(impact_tmp==0 and (first_ankle_center_x < right_wrist) ): #오차값 5부여 (x좌표가 동일하지 않는 경우가 존재
#        impact_tmp = impact_tmp + 1
#        Time['address'] = current_time
#    return Time['address'],impact_tmp

