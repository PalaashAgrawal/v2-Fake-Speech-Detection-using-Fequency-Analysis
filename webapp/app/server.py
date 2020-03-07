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

#for .wav to .png conversion
import numpy
import matplotlib.pyplot as plt
import pylab
from scipy.io import wavfile
from scipy.fftpack import fft
from PIL import Image
import os

#__________________________________________________________________________________________________________

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
    #wav_data = await request.form()
    #____________________________________________________________________________
    #converting wav to image
    samplingFreq, mySound = wavfile.read(await request.form())
    mySound = mySound / (2.**15)
    signalDuration =  mySound.shape[0] / samplingFreq
    if signalDuration>5.0:
        mySound=mySound[:5*samplingFreq]
    mySoundLength = len(mySound)
    fftArray = fft(mySound)
    numUniquePoints = int(numpy.ceil((mySoundLength + 1) / 2))
    fftArray = fftArray[0:numUniquePoints]
    fftArray = abs(fftArray)
    fftArray = fftArray / float(mySoundLength)
    fftArray = fftArray **2
    if mySoundLength % 2 > 0:
        fftArray[1:len(fftArray)] = fftArray[1:len(fftArray)] * 2
    else:
        fftArray[1:len(fftArray) -1] = fftArray[1:len(fftArray) -1] * 2
    freqArray = numpy.arange(0, numUniquePoints, 1.0) * (samplingFreq / mySoundLength);
    fftArraymin=numpy.min(fftArray[numpy.nonzero(fftArray)])
    fftArray[fftArray==0]=fftArraymin
    img_data=plt.figure()
    plt.plot(freqArray/1000, 10 * numpy.log10 (fftArray), color='b')
    plt.axis('off')
    #____________________________________________________________________
        
    img_bytes = await (img_data['file'].read())
    
    img = open_image(BytesIO(img_bytes))
    prediction = learn.predict(img)[0]
    return JSONResponse({'result': str(prediction)})


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")
