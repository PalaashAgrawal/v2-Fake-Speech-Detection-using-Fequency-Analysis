import aiohttp
import asyncio
import uvicorn
from fastai import *
from fastai.vision import *
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles

#_________________________________________________________________________________________________________
#for .wav to .png conversion
import numpy
import matplotlib.pyplot as plt
import pylab
from scipy.io import wavfile
from scipy.fftpack import fft
from PIL import Image
import os
#__________________________________________________________________________________________________________
def transform(audio,samplingFreq):
    audio = audio / (2.**15)

    #Check sample points and sound channel for duel channel(5060, 2) or  (5060, ) for mono channel
    mySoundShape = audio.shape
    samplePoints = float(audio.shape[0])

    #Get duration of sound file
    signalDuration =  audio.shape[0] / samplingFreq
    
    if signalDuration>5.0:
        audio=audio[:5*samplingFreq]
    
    #If two channels, then select only one channel
    mySoundOneChannel = audio#[:,0]
    mySoundLength = len(audio)

    #Take the Fourier transformation on given sample point 
    #fftArray = fft(mySound)
    fftArray = fft(mySoundOneChannel)

    numUniquePoints = int(numpy.ceil((mySoundLength + 1) / 2))
    fftArray = fftArray[0:numUniquePoints]

    fftArray = abs(fftArray) #modulus of a complex number

    #Scale the fft array by length of sample points so that magnitude does not depend on
    #the length of the signal or on its sampling frequency

    fftArray = fftArray / float(mySoundLength)

    #FFT has both positive and negative information. Square to get positive only
    fftArray = fftArray **2

    #Multiply by two (research why?)
    #Odd NFFT excludes Nyquist point
    if mySoundLength % 2 > 0: #we've got odd number of points in fft
        fftArray[1:len(fftArray)] = fftArray[1:len(fftArray)] * 2

    else: #We've got even number of points in fft
        fftArray[1:len(fftArray) -1] = fftArray[1:len(fftArray) -1] * 2  

    freqArray = numpy.arange(0, numUniquePoints, 1.0) * (samplingFreq / mySoundLength);

    
    audioname=str(audiopath)
    #Plot the frequency
    fftArraymin=numpy.min(fftArray[numpy.nonzero(fftArray)])
    fftArray[fftArray==0]=fftArraymin
    img =plt.figure()
    plt.plot(freqArray/1000, 10 * numpy.log10 (fftArray), color='b')
    plt.xlabel('Frequency (Khz)')
    plt.ylabel('Power (dB)')
    plt.axis('off')
    return img
#___________________________________________________
export_file_url = 'https://drive.google.com/uc?export=download&id=10XQKmnv7zfAAER-PFoJIV2cpt6Hglt-1'
export_file_name = 'export.pkl'

classes=['Real-Donald-Trump','Donald-Trump-Impersonation']
path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))


async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)


async def setup_learner():
    await download_file(export_file_url, path / export_file_name)
    try:
        learn = load_learner(path, export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise


loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())


@app.route('/analyze', methods=['POST'])
async def analyze(request):
    img_data = await request.form()
    #img_bytes = await (img_data['file'].read())
    #img = open_image(BytesIO(img_bytes))

    signal_freq,data=wavfile.read(img_data)
    IMGfig=transform(data,signal_freq) #This is a figure and cannot be fed into the model. It Needs to be converted to an image first, without saving it.
   # _____________________
    #now to convert figure to image
    buffer = io.BytesIO()
    IMG.savefig(buffer, format='png', dpi = 300)
    buffer.seek(0)
    img=open_image(buffer)
    #___________________
    prediction = learn.predict(img)[0]
    buffer.close()
    return JSONResponse({'result': str(prediction)})


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")
