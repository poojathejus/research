import base64
from io import BytesIO

import cv2
import hashlib
import bitstring
import numpy
import numpy as np
import zigzag as zz
import image_preparation as imgp
import data_embedding as stego
# import run_stego_algorithm as src
import struct
from flask import Flask, render_template, request, session, send_file
from DBConnection import Db
app = Flask(__name__)
app.secret_key = "abc"
static_path="D:\\securityenhancement\\static\\"


@app.route('/login1')
def login1():
    return render_template('loginpage.html')


@app.route('/login2', methods=['POST'])
def login2():
    username = request.form['username']
    password = request.form['password']
    Dbcon = Db()
    qry = "SELECT * FROM LOGIN WHERE USERNAME='"+username+"' AND PASSWORD='"+password+"';"
    res = Dbcon.selectOne(qry)
    if res is None:
        return "<script>alert('Invalid username or password');window.location='/login1'</script>"
    else:
        session['lid'] = res['loginid']
        return render_template('adminpanel.html')


@app.route('/reg')
def reg():
    return render_template('reg.html')


@app.route('/signup1')
def signup1():
    return render_template('signup.html')


@app.route('/signup2', methods=['POST'])
def signup2():
    name = request.form['name']
    gender = request.form['gender']
    dob = request.form['dob']
    photo = request.files["photo"]
    email = request.form['email']
    phone = request.form['phone']
    password = request.form['password']
    confirmpassword = request.form['confirmpassword']

    photo.save("D:\\securityenhancement\\static\\images\\" + photo.filename)
    filename = "/static/images/" + photo.filename

    dbcon = Db()

    qry = "INSERT INTO login(USERNAME,PASSWORD)VALUES('" + email + "','" + password + "');"
    loginid = dbcon.insert(qry)

    qry2 = "INSERT INTO userreg(NAME,gender,dob,filename,email,phone,loginid)VALUES('"+name+"','"+gender+"','"+dob+"','"+filename+"','"+email+"','"+phone+"','"+str(loginid)+"');"
    dbcon.insert(qry2)

    return render_template('loginpage.html')


@app.route('/changepassword1')
def changepassword():
    return render_template('changepassword.html')


@app.route('/changepassword2', methods=['POST'])
def changepassword2():
    oldpassword = request.form['oldpassword']
    newpassword = request.form['newpassword']
    confirmpassword = request.form['confirmpassword']

    Dbcon = Db()
    qry1 = "SELECT * FROM login WHERE loginid='" + str(session['lid']) + "' AND PASSWORD='"+oldpassword+"';"
    res = Dbcon.selectOne(qry1)
    if res is not None:
        qry2 = "UPDATE login SET PASSWORD='"+confirmpassword+"' WHERE loginid='"+str(session['lid'])+"';"
        Dbcon.update(qry2)
        return "<script>alert('Password Changed');window.location='/login1'</script>"
    else:
        return "<script>alert('Incorrect password');window.location='/login1'</script>"


@app.route('/viewprofile1')
def viewprofile1():
    Dbcon = Db()
    qry = "SELECT * FROM userreg WHERE loginid='"+str(session['lid'])+"';"
    res = Dbcon.selectOne(qry)
    return render_template('viewprofile.html', data=res)


@app.route('/encryptionphase1a')
def encryptionphase1a():
    return render_template('encryptionphase1.html')


@app.route('/encryptionphase1b', methods=['POST'])
def encryptionphase1b():
    from datetime import datetime
    a = str(datetime.now().year) + str(datetime.now().month) + str(datetime.now().day) + str(datetime.now().hour) + str(datetime.now().minute) + str(datetime.now().second)

    image = request.files["image"]
    image.save("D:\\securityenhancement\\static\\images\\"+a+image.filename)
    coverimage = request.files["coverimage"]
    coverimage.save("D:\\securityenhancement\\static\\images\\"+coverimage.filename)

    session['originalimage'] = a+image.filename
    session['coverimage'] = coverimage.filename

    return "<script>window.location='/agf'</script>"


@app.route('/encryptionphase2a')
def encryptionphase2a():

    return render_template('encryptionphase2.html')


@app.route('/encryptionphase3a')
def encryptionphase3a():
    return render_template('encryptionphase3.html')


@app.route('/myfiles1')
def myfiles1():
    dbcon = Db()
    qry = "SELECT * FROM uploads WHERE userid='"+str(session['lid'])+"';"
    res = dbcon.select(qry)
    return render_template('myfiles.html', data=res)


@app.route('/myfiles2', methods=['POST'])
def myfiles2():
    button1 = request.form['button']
    if button1 == "Filter":
        datefrom = request.form['from']
        dateto = request.form['to']

        dbcon = Db()
        qry = "SELECT * FROM uploads WHERE userid='" + str(session['lid']) + "' and date between '"+datefrom+"' and '"+dateto+"';"
        res = dbcon.select(qry)
        return render_template('myfiles.html', data=res)

    if button1 == 'Search':
        textvalue = request.form['text']

        dbcon = Db()
        qry = "SELECT * FROM uploads WHERE userid='" + str(session['lid']) + "' and filename like '%"+textvalue+"%';"
        res = dbcon.select(qry)
        return render_template('myfiles.html', data=res)

    return render_template('myfiles.html')


@app.route('/downloadphase1/<filename>/<filename2>/<fid>')
def downloadphase1(filename, filename2, fid):

    # print(filename)
    # print(filename2)
    path = "D:\\securityenhancement\\static\\dcthiddenimage\\"+filename
    path2 = "D:\\securityenhancement\\static\\temp\\" + filename2
    # print(path)
    # print(path2)
    from decode import maindecode
    path3 = "D:\\securityenhancement\\static\\temp\\final.bmp"
    maindecode(path2, path3)

    stego_image = cv2.imread(path3, flags=cv2.IMREAD_COLOR)
    stego_image_f32 = np.float32(stego_image)
    stego_image_YCC = imgp.YCC_Image(cv2.cvtColor(stego_image_f32, cv2.COLOR_BGR2YCrCb))

    # FORWARD DCT STAGE
    dct_blocks = [cv2.dct(block) for block in stego_image_YCC.channels[0]]  # Only care about Luminance layer

    # QUANTIZATION STAGE
    dct_quants = [np.around(np.divide(item, imgp.JPEG_STD_LUM_QUANT_TABLE)) for item in dct_blocks]

    # Sort DCT coefficients by frequency
    sorted_coefficients = [zz.zigzag(block) for block in dct_quants]

    # DATA EXTRACTION STAGE
    recovered_data = stego.extract_encoded_data_from_DCT(sorted_coefficients)

    # Determine length of secret message
    data_len = int(recovered_data.read('uint:32') / 8)
    # print(data_len)
    # print("hello world")

    # Extract secret message from DCT coefficients
    extracted_data = bytes()
    for _ in range(data_len):
        try:
            extracted_data += struct.pack('>B', recovered_data.read('uint:8'))
            print("hai")

        except:
            print("error")

    print(extracted_data)
    print(type(extracted_data))
    print(len(extracted_data))
    b = base64.b64encode(extracted_data)
    c = b.hex()
    # print("how are you")
    print(c)
    # print("aaaaaaaaaaaaaaaaa")
    # print(b)
    # print(type(b))
    # print(len(b), "length of b")
    # hashnew = extracted_data.decode('ascii')
    a = "SELECT hash FROM uploads WHERE fileid='"+fid+"';"
    Dbcon = Db()
    res = Dbcon.selectOne(a)
    h = res['hash']
    b = bytes.fromhex(h)
    print(b,"  new byte")
    print(len(b))
    print(h)
    success_counter=0
    false_counter=0
    for i in range(0,32):
        if b[i]==extracted_data[i]:
            success_counter = success_counter+1
        else:
            false_counter =false_counter+1
    print(success_counter , " successs couner")
    print(false_counter  , " false counter")


    print(type(b),type(extracted_data))

    PSNR_val=find_psnr(b, extracted_data)
    print("PSNR  ", PSNR_val)

    cd= float(success_counter)/float(32)

    if(success_counter > 30):

        session["p"]=path3
        return render_template("ad.html",a=PSNR_val[0],b=PSNR_val[1],c=path3,cd=cd)

        # return send_file(path3, as_attachment=True)
    else:
        session["p"] = path3
        return "<script>alert('file alteration found');window.location='/myfiles1'</script>"

    # Print secret message back to the user
    # print(extracted_data.decode('ascii'))
    # return render_template('downloadphase.html')


@app.route("/download")
def download():
    return send_file(str(session["p"]),as_attachment=True)



@app.route('/adminhome')
def adminhome():
    return render_template('home.html')


@app.route('/adminpanel')
def adminpanel():
    return render_template('adminpanel.html')


@app.route('/loginnew1')
def loginnew1():
    return render_template('loginnew.html')


@app.route('/editprofile1')
def editprofile1():
    Dbcon = Db()
    qry = "SELECT * FROM userreg WHERE loginid='" + str(session['lid']) + "';"
    res = Dbcon.selectOne(qry)
    return render_template('editprofile.html', data=res)


@app.route('/editprofile2', methods=['POST'])
def editprofile2():
    Dbcon = Db()

    name = request.form['name']
    gender = request.form['gender']
    dob = request.form['dob']
    email = request.form['email']
    phone = request.form['phone']
    if 'photo' in request.files:
        photo = request.files['photo']
        print("hello",photo.filename)
        print(len(photo.filename), "length of photo")
        if photo.filename!="":
            photo.save("D:\\securityenhancement\\static\\images\\" + photo.filename)
            filename = "/static/images/" + photo.filename
            qrey = "UPDATE userreg SET NAME='"+name+"',gender='"+gender+"',dob='"+dob+"',filename='"+filename+"',email='"+email+"',phone='"+phone+"' WHERE loginid='"+str(session['lid'])+"';"
            Dbcon.update(qrey)
        else:
            qrey = "UPDATE userreg SET NAME='" + name + "',gender='" + gender + "',dob='" + dob + "',email='" + email + "',phone='" + phone + "' WHERE loginid='" + str(
                session['lid']) + "';"
            Dbcon.update(qrey)

    else:
        qrey = "UPDATE userreg SET NAME='" + name + "',gender='" + gender + "',dob='" + dob + "',email='" + email + "',phone='" + phone + "' WHERE loginid='" + str(
            session['lid']) + "';"
        Dbcon.update(qrey)

    return render_template('adminpanel.html')


@app.route('/agf')
def agf():
    originalimage = session['originalimage']
    coverimage = session['coverimage']
    return render_template('agf.html', originalimage=originalimage, coverimage=coverimage)


@app.route('/agf2', methods=['POST'])
def agf2():
    x = int(request.form['x'])
    y = int(request.form['y'])
    w = int(request.form['w'])
    h = int(request.form['h'])
    # print(x, y, w, h)
    originalimage = session['originalimage']
    p = "D:\\securityenhancement\\static\\images\\"+originalimage
    img = cv2.imread(p)
    # print(img.shape)
    print(p, "original image file path")
    # print(img.shape)
    imgcrop = img[y:y+w,x:x+h]
    #print(x+h)
    #print(y+w)
    #print(x)
    #print(y)
    cv2.imwrite("D:\\securityenhancement\\static\\images\\a.jpg", imgcrop)

    filename = "D:\\securityenhancement\\static\\images\\a.jpg"
    sha256_hash = hashlib.sha256()
    with open(filename, "rb") as f:
        # Read and update hash string value in blocks of 4K
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
        print(sha256_hash.hexdigest())
        # print("how are you......??")

        NUM_CHANNELS = 3
        originalimage = session['originalimage']
        path1 = "D:\\securityenhancement\\static\\images\\"+originalimage
        COVER_IMAGE_FILEPATH = path1
        # Choose your cover image (PNG)
        # print(COVER_IMAGE_FILEPATH)
        from datetime import datetime
        a = str(datetime.now().year)+str(datetime.now().month)+str(datetime.now().day)+str(datetime.now().hour)+str(datetime.now().minute)+str(datetime.now().second)
        STEGO_IMAGE_FILEPATH = "D:\\securityenhancement\\static\\dcthiddenimage\\" +a+".png"
        # SECRET_MESSAGE_STRING = sha256_hash.hexdigest()
        SECRET_MESSAGE_STRING = base64.b64encode(bytes.fromhex(sha256_hash.hexdigest()))
        # ============================================================================= #
        # ============================================================================= #
        # =========================== BEGIN CODE OPERATION ============================ #
        # ============================================================================= #
        # ============================================================================= #

        raw_cover_image = cv2.imread(COVER_IMAGE_FILEPATH, flags=cv2.IMREAD_COLOR)
        height, width = raw_cover_image.shape[:2]
        # Force Image Dimensions to be 8x8 compliant
        while (height % 8): height += 1  # Rows
        while (width % 8): width += 1  # Cols
        valid_dim = (width, height)
        padded_image = cv2.resize(raw_cover_image, valid_dim)
        cover_image_f32 = np.float32(padded_image)
        cover_image_YCC = imgp.YCC_Image(cv2.cvtColor(cover_image_f32, cv2.COLOR_BGR2YCrCb))

        # Placeholder for holding stego image data
        stego_image = np.empty_like(cover_image_f32)

        for chan_index in range(NUM_CHANNELS):
            # FORWARD DCT STAGE
            dct_blocks = [cv2.dct(block) for block in cover_image_YCC.channels[chan_index]]

            # QUANTIZATION STAGE
            dct_quants = [np.around(np.divide(item, imgp.JPEG_STD_LUM_QUANT_TABLE)) for item in dct_blocks]

            # Sort DCT coefficients by frequency
            sorted_coefficients = [zz.zigzag(block) for block in dct_quants]

            # Embed data in Luminance layer
            if (chan_index == 0):
                # DATA INSERTION STAGE
                secret_data = ""
                ms = base64.b64decode(SECRET_MESSAGE_STRING)
                # print(ms)
                # print("welcome")
                for char in ms: secret_data += bitstring.pack('uint:8', char)
                embedded_dct_blocks, status = stego.embed_encoded_data_into_DCT(secret_data, sorted_coefficients)
                if status == False:
                    return "<script>alert('cannot use this image')window.location='/agf'</script>"

                desorted_coefficients = [zz.inverse_zigzag(block, vmax=8, hmax=8) for block in embedded_dct_blocks]
            else:
                # Reorder coefficients to how they originally were
                desorted_coefficients = [zz.inverse_zigzag(block, vmax=8, hmax=8) for block in sorted_coefficients]

            # DEQUANTIZATION STAGE
            dct_dequants = [np.multiply(data, imgp.JPEG_STD_LUM_QUANT_TABLE) for data in desorted_coefficients]

            # Inverse DCT Stage
            idct_blocks = [cv2.idct(block) for block in dct_dequants]

            # Rebuild full image channel
            stego_image[:, :, chan_index] = np.asarray(
                imgp.stitch_8x8_blocks_back_together(cover_image_YCC.width, idct_blocks))
        # -------------------------------------------------------------------------------------------------------------------#

        # Convert back to RGB (BGR) Colorspace
        stego_image_BGR = cv2.cvtColor(stego_image, cv2.COLOR_YCR_CB2BGR)

        # Clamp Pixel Values to [0 - 255]
        final_stego_image = np.uint8(np.clip(stego_image_BGR, 0, 255))

        # Write stego image
        cv2.imwrite(STEGO_IMAGE_FILEPATH, final_stego_image)

        a = "D:\\securityenhancement\\static\\images\\"+session['coverimage']
        # print("a", a)

        from encode import mainencode
        from datetime import datetime
        path = str(datetime.now().year)+str(datetime.now().month)+str(datetime.now().day)+str(datetime.now().hour)+str(datetime.now().minute)+str(datetime.now().second)

        v = "D:\\securityenhancement\\static\\temp\\"+path+".bmp"
        # print(v)
        mainencode(STEGO_IMAGE_FILEPATH, v, a)
        print(SECRET_MESSAGE_STRING)
        print(type(SECRET_MESSAGE_STRING))
        print(len(SECRET_MESSAGE_STRING),"length of secret message string")
        b = base64.b64encode(SECRET_MESSAGE_STRING)
        c = b.hex()
        # print(len(b),"length of b")
        # print(len(c),"length of c")
        print(c)
        # print('HAI')
        # print(type(SECRET_MESSAGE_STRING))
        SECRET_MESSAGE_STRING=sha256_hash.hexdigest()
        dbcon = Db()

        qry1 = "INSERT INTO uploads (title,filename,DATE,roi,HASH,userid,filename2) VALUES ('title3','"+a+".png"+"', now(),'"+str(x)+","+str(y)+","+str(w)+","+str(h)+"','"+SECRET_MESSAGE_STRING+"','" + str(
            session['lid']) + "','"+path+".bmp"+"');"
        dbcon.insert(qry1)



    return render_template('encryptionphase2.html', originalimage=originalimage)


@app.route('/index')
def index():
    return render_template('loginnew.html')

@app.route("/two_files")
def two_files():
    return render_template("two_files.html")

###################         PSNR
import math
def find_psnr(img1, img2):

    print(len(img1))
    print(len(img2))

    a= [x for x in img1]
    b= [x for x in img2]
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

    a=numpy.array(a)
    b=numpy.array(b)


    # np_Arr1=np_bytes1 = BytesIO(img1)
    # np_Arr2=np_bytes2 = BytesIO(img2)

    mse = np.mean((a - b) ** 2)
    mse = np.mean((a - b) ** 2)

    print(a)
    print(b)

    print(mse)
    if mse == 0:
        return 100
    PIXEL_MAX = max(max(a),max(b))
    print(PIXEL_MAX)
    print(20 * math.log10(PIXEL_MAX / math.sqrt(mse)),"Hoooo")
    # print(20 * math.log10(PIXEL_MAX) / mse ^ ,"Hoooo")
    return 20 * math.log10(PIXEL_MAX)/ math.sqrt(mse), mse
    # return 10 * math.log10(1. / mse)
    pass
#################################

@app.route("/two_files_post", methods=['post'])
def two_files_post():
    file1=request.files['file1']
    file2=request.files['file2']
    file1.save(static_path+"two_img\\a.jpg")
    file2.save(static_path+"two_img\\b.jpg")
    img1=cv2.imread(static_path+"two_img\\a.jpg")
    img2=cv2.imread(static_path+"two_img\\b.jpg")
    psnr = find_psnr(img1, img2)
    print(psnr)
    # sm=0
    # print("Hiii  ", img1.shape)
    # height=img1.shape[0]
    # width=img1.shape[1]
    # for i in range(0, height):
    #     for j in range(0, width):
    #         R1, G1, B1=img1[i, j]
    #         R2, G2, B2=img2[i, j]
    #         C1=max(R1,R2)-min(R1,R2)
    #         C2=max(G1,G2)-min(G1,G2)
    #         C3=max(B1,B2)-min(B1,B2)
    #         sm=sm + (C1 + C2 + C3)
    # new_val=sm/(height * width * 3)
    # print(new_val)

    return "ok"

if __name__=='__main__':
    app.run(port=4000)