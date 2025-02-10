#-----------------------------------------------------------------------------#
#-----------------------Quick Start Guide for Python--------------------------#
#-----------------------------------------------------------------------------#
#------------------QBot Platform with Mobile Robotics Lab----------------------#
#-----------------------------------------------------------------------------#

# Imports
import os
import sys
import platform
import cv2
import numpy as np
import platform
import time
from quanser.devices import (
    RangingMeasurements,
    RangingMeasurementMode,
    DeviceError,
    RangingDistance
)
from quanser.multimedia import Video3D, Video3DStreamType, VideoCapture, \
    MediaError, ImageFormat, ImageDataType, VideoCapturePropertyCode, \
    VideoCaptureAttribute
from quanser.communications import Stream, StreamError, PollFlag
try:
    from quanser.common import Timeout
except:
    from quanser.communications import Timeout

from qlabs_setup import setup


class BasicStream:
    '''Class object consisting of basic stream server/client functionality'''
    def __init__(self, uri, agent='S', receiveBuffer=np.zeros(1, dtype=np.float64), sendBufferSize=2048, recvBufferSize=2048, nonBlocking=False, verbose=False):
        '''
        This functions simplifies functionality of the quanser_stream module to provide a
        simple server or client. \n
         \n
        uri - IP server and port in one string, eg. 'tcpip://IP_ADDRESS:PORT' \n
        agent - 'S' or 'C' string representing server or client respectively \n
        receiveBuffer - numpy buffer that will be used to determine the shape and size of data received \n
        sendBufferSize - (optional) size of send buffer, default is 2048 \n
        recvBufferSize - (optional) size of recv buffer, default is 2048 \n
        nonBlocking - set to False for blocking, or True for non-blocking connections \n
         \n
        Stream Server as an example running at IP 192.168.2.4 which receives two doubles from the client: \n
        >>> myServer = BasicStream('tcpip://localhost:18925', 'S', receiveBuffer=np.zeros((2, 1), dtype=np.float64))
         \n
        Stream Client as an example running at IP 192.168.2.7 which receives a 480 x 640 color image from the server: \n
        >>> myClient = BasicStream('tcpip://192.168.2.4:18925', 'C', receiveBuffer=np.zeros((480, 640, 3), dtype=np.uint8))

        '''
        self.agent 			= agent
        self.sendBufferSize = sendBufferSize
        self.recvBufferSize = recvBufferSize
        self.uri 			= uri
        self.receiveBuffer  = receiveBuffer
        self.verbose        = verbose
        # If the agent is a Client, then Server isn't needed.
        # If the agent is a Server, a Client will also be needed. The server can start listening immediately.

        self.clientStream = Stream()
        if agent=='S':
            self.serverStream = Stream()

        # Set polling timeout to 10 milliseconds
        self.t_out = Timeout(seconds=0, nanoseconds=10000000)

        # connected flag initialized to False
        self.connected = False

        try:
            if agent == 'C':
                self.connected = self.clientStream.connect(uri, nonBlocking, self.sendBufferSize, self.recvBufferSize)
                if self.connected and self.verbose:
                    print('Connected to a Server successfully.')

            elif agent == 'S':
                if self.verbose:
                    print('Listening for incoming connections.')
                self.serverStream.listen(self.uri, nonBlocking)
            pass

        except StreamError as e:
            if self.agent == 'S' and self.verbose:
                print('Server initialization failed.')
            elif self.agent == 'C' and self.verbose:
                print('Client initialization failed.')
            print(e.get_error_message())

    def checkConnection(self, timeout=Timeout(seconds=0, nanoseconds=100)):
        '''When using non-blocking connections (nonBlocking set to True), the constructor method for this class does not block when
        listening (as a server) or connecting (as a client). In such cases, use the checkConnection method to attempt continuing to
        accept incoming connections (as a server) or connect to a server (as a client).  \n
         \n
        Stream Server as an example \n
        >>> while True:
        >>> 	if not myServer.connected:
        >>> 		myServer.checkConnection()
        >>>		if myServer.connected:
        >>> 		yourCodeGoesHere()
         \n
        Stream Client as an example \n
        >>> while True:
        >>> 	if not myClient.connected:
        >>> 		myClient.checkConnection()
        >>>		if myClient.connected:
        >>> 		yourCodeGoesHere()
         \n
        '''
        if self.agent == 'C' and not self.connected:
            try:
                pollResult = self.clientStream.poll(timeout, PollFlag.CONNECT)

                if (pollResult & PollFlag.CONNECT) == PollFlag.CONNECT:
                    self.connected = True
                    if self.verbose: print('Connected to a Server successfully.')

            except StreamError as e:
                if e.error_code == -33:
                    self.connected = self.clientStream.connect(self.uri, True, self.sendBufferSize, self.recvBufferSize)
                else:
                    if self.verbose: print('Client initialization failed.')
                    print(e.get_error_message())

        if self.agent == 'S' and not self.connected:
            try:
                pollResult = self.serverStream.poll(self.t_out, PollFlag.ACCEPT)
                if (pollResult & PollFlag.ACCEPT) == PollFlag.ACCEPT:
                    self.connected = True
                    if self.verbose: print('Found a Client successfully...')
                    self.clientStream = self.serverStream.accept(self.sendBufferSize, self.recvBufferSize)

            except StreamError as e:
                if self.verbose: print('Server initialization failed...')
                print(e.get_error_message())

    def terminate(self):
        '''Use this method to correctly shutdown and then close connections. This method automatically closes all streams involved (Server will shutdown server streams as well as client streams). \n
         \n
        Stream Server as an example \n
        >>> while True:
        >>> 	if not myServer.connected:
        >>> 		myServer.checkConnection()
        >>>		if myServer.connected:
        >>> 		yourCodeGoesHere()
        >>>			if breakCondition:
        >>>				break
        >>> myServer.terminate()
         \n
        Stream Client as an example	 \n
        >>> while True:
        >>> 	if not myClient.connected:
        >>> 		myClient.checkConnection()
        >>>		if myClient.connected:
        >>> 		yourCodeGoesHere()
        >>>			if breakCondition:
        >>>				break
        >>> myClient.terminate()

        '''

        if self.connected:
            self.clientStream.shutdown()
            self.clientStream.close()
            if self.verbose: print('Successfully terminated clients...')

        if self.agent == 'S':
            self.serverStream.shutdown()
            self.serverStream.close()
            if self.verbose: print('Successfully terminated servers...')

    def receive(self, iterations=1, timeout=Timeout(seconds=0, nanoseconds=10)):
        '''
        This functions populates the receiveBuffer with bytes if available. \n \n

        Accepts: \n
        iterations - (optional) number of times to poll for incoming data before terminating, default is 1 \n
         \n
        Returns: \n
        receiveFlag - flag indicating whether the number of bytes received matches the expectation. To check the actual number of bytes received, use the bytesReceived class object. \n
         \n
        Stream Server as an example \n
        >>> while True:
        >>> 	if not myServer.connected:
        >>> 		myServer.checkConnection()
        >>>		if myServer.connected:
        >>> 		flag = myServer.receive()
        >>>			if breakCondition or not flag:
        >>>				break
        >>> myServer.terminate()
         \n
        Stream Client as an example	 \n
        >>> while True:
        >>> 	if not myClient.connected:
        >>> 		myClient.checkConnection()
        >>>		if myClient.connected:
        >>> 		flag = myServer.receive()
        >>>			if breakCondition or not flag:
        >>>				break
        >>> myClient.terminate()

        '''

        self.t_out = timeout
        counter = 0
        dataShape = self.receiveBuffer.shape

        # Find number of bytes per array cell based on type
        numBytesBasedOnType = len(np.array([0], dtype=self.receiveBuffer.dtype).tobytes())

        # Calculate total dimensions
        dim = 1
        for i in range(len(dataShape)):
            dim = dim*dataShape[i]

        # Calculate total number of bytes needed and set up the bytearray to receive that
        totalNumBytes = dim*numBytesBasedOnType
        self.data = bytearray(totalNumBytes)
        self.bytesReceived = 0
        # print(totalNumBytes)
        # Poll to see if data is incoming, and if so, receive it. Poll a max of 'iteration' times
        try:
            while True:

                # See if data is available
                pollResult = self.clientStream.poll(self.t_out, PollFlag.RECEIVE)
                counter += 1
                if not (iterations == 'Inf'):
                    if counter > iterations:
                        break
                if not ((pollResult & PollFlag.RECEIVE) == PollFlag.RECEIVE):
                    continue # Data not available, skip receiving

                # Receive data
                self.bytesReceived = self.clientStream.receive_byte_array(self.data, totalNumBytes)

                # data received, so break this loop
                break

            #  convert byte array back into numpy array and reshape.
            self.receiveBuffer = np.reshape(np.frombuffer(self.data, dtype=self.receiveBuffer.dtype), dataShape)

        except StreamError as e:
            print(e.get_error_message())
        finally:
            receiveFlag = self.bytesReceived==1
            return receiveFlag, totalNumBytes*self.bytesReceived

    def send(self, buffer):
        """
        This functions sends the data in the numpy array buffer
        (server or client). \n \n

        INPUTS: \n
        buffer - numpy array of data to be sent \n

        OUTPUTS: \n
        bytesSent - number of bytes actually sent (-1 if send failed) \n
         \n
        Stream Server as an example \n
        >>> while True:
        >>> 	if not myServer.connected:
        >>> 		myServer.checkConnection()
        >>>		if myServer.connected:
        >>> 		sent = myServer.send()
        >>>			if breakCondition or sent == -1:
        >>>				break
        >>> myServer.terminate()
         \n
        Stream Client as an example	 \n
        >>> while True:
        >>> 	if not myClient.connected:
        >>> 		myClient.checkConnection()
        >>>		if myClient.connected:
        >>> 		sent = myServer.send()
        >>>			if breakCondition or sent == -1:
        >>>				break
        >>> myClient.terminate()

        """

        # Set up array to hold bytes to be sent
        byteArray = buffer.tobytes()
        self.bytesSent = 0

        # Send bytes and flush immediately after
        try:
            self.bytesSent = self.clientStream.send_byte_array(byteArray, len(byteArray))
            self.clientStream.flush()
        except StreamError as e:
            print(e.get_error_message())
            self.bytesSent = -1 # If an error occurs, set bytesSent to -1 for user to check
        finally:
            return self.bytesSent



IS_PHYSICAL_QBOTPLATFORM = (('nvidia' == os.getlogin())
                            and ('aarch64' == platform.machine()))
"""A boolean constant indicating if the current device is a physical QBot
Platform.

This constant is set to True if both the following conditions are met:
1. The current user's login name is 'nvidia'.
2. The underlying system's hardware architecture is 'aarch64'.

It's intended to be used for configuring execution as needed depending on if
the executing platform is a physical and virtual QBot Platform.
"""

class QBotPlatformDriver():
    """Driver class for performing basic QBot Platform IO

    Args:
            mode (int, optional): Determines the driver mode. Defaults to 1.
                1 & 2 are designed for education, 3 & 4 are designed for
                research. 1 and 3 are body mode, 2 and 4 are wheeled mode.
            ip (str): IP address of the QBot Platform.
    """

    def __init__(self, mode=1, ip='192.168.2.15') -> None:

        # QBot reads
        self.wheelPositions = np.zeros((2), dtype = np.float64)
        self.wheelSpeeds    = np.zeros((2), dtype = np.float64)
        self.motorCmd       = np.zeros((2), dtype = np.float64)
        self.accelerometer  = np.zeros((3), dtype = np.float64)
        self.gyroscope      = np.zeros((3), dtype = np.float64)
        self.currents       = np.zeros((2), dtype = np.float64)
        self.battVoltage    = np.zeros((1), dtype = np.float64)
        self.watchdog       = np.zeros((1), dtype = np.float64)

        # QBot Platform Driver listening on port 18888
        self.uri = 'tcpip://'+ip+':18888'

        # 1 ms timeout parameter
        self._timeout = Timeout(seconds=0, nanoseconds=1000000)

        # establish stream object to communicate with QBot Platform Driver
        self._handle = BasicStream(uri=self.uri,
                                    agent='C',
                                    receiveBuffer=np.zeros((17),
                                                           dtype=np.float64),
                                    sendBufferSize=2048,
                                    recvBufferSize=2048,
                                    nonBlocking=True)

        # Only set mode on initialization, this value is not set in read-write
        self._sendPacket = np.zeros((10), dtype=np.float64)
        self._sendPacket[0] = mode
        self._mode = mode

        # if connected to the Driver, proceed, else, try to connect.
        # self.status_check('Connected to QBot Platform Driver.', iterations=20)
        self.status_check('', iterations=20)
        # there is no return here.

    def status_check(self, message, iterations=10):
        # blocking method to establish connection to the server stream.
        self._timeout = Timeout(seconds=0, nanoseconds=1000) #1000000
        counter = 0
        while not self._handle.connected:
            self._handle.checkConnection(timeout=self._timeout)
            counter += 1
            if self._handle.connected:
                print(message)
                break
            elif counter >= iterations:
                print('Driver error: status check failed.')
                break

            # once you connect, self._handle.connected goes True, and you
            # leave this loop.

    def read_write_std(self,
                       timestamp,
                       arm = 1,
                       commands=np.zeros((2), dtype=np.float64),
                       userLED=False,
                       color=[1, 0, 1],
                       hold = 0):

        # data received flag
        new = False

        # 1 us timeout parameter
        self._timeout = Timeout(seconds=0, nanoseconds=10000000)

        # set User LED color values if desired
        if userLED:
            self._sendPacket[1] = 1.0 # User LED packet
            self._sendPacket[2:5] = np.array([color[0], color[1], color[2]])
        else:
            self._sendPacket[1] = 0.0 # User LED packet
            self._sendPacket[2:5] = np.array([0, 0, 0])

        # set remaining packet to send
        self._sendPacket[5] = arm
        self._sendPacket[6] = hold
        self._sendPacket[7] = commands[0]
        self._sendPacket[8] = commands[1]
        self._sendPacket[9] = timestamp

        # if connected to driver, send/receive
        if self._handle.connected:
            self._handle.send(self._sendPacket)
            new, bytesReceived = self._handle.receive(timeout=self._timeout, iterations=5)
            # print(new, bytesReceived)
            # if new is True, full packet was received
            if new:
                self.wheelPositions = self._handle.receiveBuffer[0:2]
                self.wheelSpeeds = self._handle.receiveBuffer[2:4]
                self.motorCmd = self._handle.receiveBuffer[4:6]
                self.accelerometer = self._handle.receiveBuffer[6:9]
                self.gyroscope = self._handle.receiveBuffer[9:12]
                self.currents = self._handle.receiveBuffer[12:14]
                self.battVoltage = self._handle.receiveBuffer[14]
                self.watchdog = self._handle.receiveBuffer[15]
                self.timeStampRecv = self._handle.receiveBuffer[16]

        else:
            self.status_check('Reconnected to QBot Platform Driver.')

        # if new is False, data is stale, else all is good
        return new

    def terminate(self):
        self._handle.terminate()

# Section A - Setup
actor = setup(locationQBotP=[0, 0, 0], verbose=True)
time.sleep(2)
ipHost, ipDriver = 'localhost', 'localhost'
commands, arm, noKill = np.zeros((2), dtype = np.float64), 0, True
frameRate, sampleRate = 60.0, 1/5.0
endFlag, counter, counterRS, counterDown, counterLidar = False, 0, 0, 0, 0
simulationTime = 30.0
startTime = time.time()
def elapsed_time():
    return time.time() - startTime

px = [0]
py = [0]
theta = [0]
pose = np.array([[0,0,0]])
myQBot       = QBotPlatformDriver(mode=3, ip=ipDriver)
startTime = time.time()
# Execute main loop until the elapsed_time has crossed simulationTime 
while elapsed_time() < simulationTime: 
    # Measured the current timestamp 
    start = elapsed_time() 
    print(elapsed_time())
    k = actor.command_and_request_state(0,0,leftLED=[1, 0, 0], rightLED=[1, 0, 0])
    # print(k[1])
    px = np.append(px, k[1][0])
    py = np.append(py, -k[1][1])
    dx = np.diff(px)
    dy = np.diff(py)
    theta = np.arctan2(dy, dx)
    print(theta[-1])
    print(k[8])

    commands, arm, noKill = [0.1,0.05], 1, True
    newHIL = myQBot.read_write_std(timestamp = time.time() - startTime,
                                        arm = arm,
                                        commands = commands)
    # Measure the last timestamp 
    end = elapsed_time() 
 
    # Calculate the computation time of your code per iteration 
    computationTime = end - start  
    # print(computationTime)
    # If the computationTime is greater than or equal  
    # to sampleTime, proceed onto next step 
    if computationTime < sampleRate: 
        # sleep for the remaining time left in this iteration 
        time.sleep(sampleRate - computationTime)  