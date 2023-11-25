from calculate_angle import calculate_angle

def address(first_ankle_center_x,landmarks_dict,Time,current_time,image_width,address_tmp):
    right_wrist = (landmarks_dict['left_wrist'][0] *image_width)
    if(address_tmp==0 and (first_ankle_center_x > right_wrist) ): #오차값 5부여 (x좌표가 동일하지 않는 경우가 존재
        address_tmp = address_tmp + 1
        Time['address'] = current_time
        print('address',Time['address'])
    return Time['address'],address_tmp


def backswing(first_right_shoulder_y,landmarks_dict,Time,current_time,image_height,back_tmp):
    left_pinky = (landmarks_dict['left_pinky'][1] *image_height)
    if(back_tmp==0 and (first_right_shoulder_y >= left_pinky) ): #오차값 5부여 (x좌표가 동일하지 않는 경우가 존재
        back_tmp = back_tmp + 1
        Time['back'] = current_time
        print('back',Time['back'])
    return Time['back'],back_tmp

def top(first_right_eye_inner_y, landmarks_dict, Time, current_time, image_height,top_tmp):
    right_wrist = (landmarks_dict['right_wrist'][1] * image_height)
    if(top_tmp==0 and (first_right_eye_inner_y - 50 > right_wrist) ): #눈썹보다 작아지게 되면
        top_tmp = top_tmp + 1
        Time['back_top'] = current_time
        print('back_top', Time['back_top'])
    return Time['back_top'],top_tmp

def impact(first_ankle_center_x,landmarks_dict,Time,current_time,image_width,impact_tmp,top_tmp):
    right_wrist = (landmarks_dict['right_wrist'][0] *image_width)
    if(Time['back'] != -1 and impact_tmp==0 and(first_ankle_center_x < right_wrist) ): #오차값 5부여 (x좌표가 동일하지 않는 경우가 존재
        impact_tmp = impact_tmp + 1
        top_tmp = top_tmp + 1
        Time['impact'] = current_time
        print('impact', Time['impact'])
    return Time['impact'],impact_tmp

def finish(current_time,total_time,Time,finish_tmp,landmarks_dict,image_width):
    #right_shoulder = (landmarks_dict['right_shoulder'][0] * image_width)
    #left_shoulder = (landmarks_dict['left_shoulder'][0] * image_width)
    right_hip = (landmarks_dict['right_hip'][0] * image_width)
    left_hip = (landmarks_dict['left_hip'][0] * image_width)
    #shoulder = abs(right_shoulder - left_shoulder)
    hip = abs(right_hip - left_hip)
    print(int(hip))
    if(finish_tmp==0 and 0<=int(hip) and int(hip)<=1):
        finish_tmp = finish_tmp + 1
        Time['finish'] = (current_time)
        print('finish', Time['finish'])
    return Time['finish'],finish_tmp
#def finish(current_time,total_time,Time,finish_tmp):
#    if(finish_tmp==0 and current_time >= int(total_time-0.5)):
#        finish_tmp = finish_tmp + 1
#        Time['finish'] = (current_time)
#        print('finish', Time['finish'])
#    return Time['finish'],finish_tmp




