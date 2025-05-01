import websockets
import json
import asyncio

class MBIS_vtube:
    def __init__(self, plugin_name="MyBitchIsAi", plugin_developer='august', port=8001):
        self.websocket = None
        self.port = port
        self.plugin_name = plugin_name
        self.plugin_developer = plugin_developer 
        self.auth_token = None

        self.msg_template = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": "TestInit",
            "messageType": "APIStateRequest"
        }
        asyncio.run(self.auth())
        print('Authenticated with VTube Studio...')
        print(f'Authentication token: {self.auth_token}')

    async def auth(self, noreturn=False) -> dict:
        if self.auth_token is None:
            msg = {
                "pluginName": self.plugin_name,
                "pluginDeveloper": self.plugin_developer
            }
            result = await self.send("AuthenticationTokenRequest", msg)
            self.auth_token = result['data']['authenticationToken']
            if noreturn:
                return
            return result

    async def send(self, msg_type: str, msg: dict, noreturn=False) -> dict:
        try:
            async with websockets.connect(f"ws://localhost:{self.port}") as websocket:
                self.websocket = websocket
                if self.auth_token and msg_type != "AuthenticationRequest":
                    auth_msg = {
                        "pluginName": self.plugin_name,
                        "pluginDeveloper": self.plugin_developer,
                        "authenticationToken": self.auth_token,
                    }
                    await self.send("AuthenticationRequest", auth_msg, noreturn=noreturn)

                msg_temp = self.msg_template.copy()
                msg_temp['messageType'] = msg_type
                if msg is not None:
                    msg_temp['data'] = msg

                await self.websocket.send(json.dumps(msg_temp))

                if noreturn:
                    return

                response = await self.websocket.recv()
                return json.loads(response)

        except websockets.exceptions.ConnectionClosedOK as e:
            print(f"Connection closed: {e}")
            # Optionally, implement reconnection logic here
            return None

        except Exception as e:
            print("Error:", e)
            print("This usually means that VTube Studio is not running or the public API is not enabled.")
            print("The permissions are not set correctly and/or got rejected.")
            raise Exception("Please start VTube Studio and enable the public API in the settings.")


    async def recv(self):
        return await self.websocket.recv()

    async def close(self):
        await self.websocket.close()

class Char_control(MBIS_vtube):
    def __init__(self, plugin_name="MyBitchIsAi", plugin_developer='HRNPH', port=8001):
        super().__init__(plugin_name, plugin_developer, port)

    async def express(self, expression: str, expression_dict=None):
        if expression_dict is None:
            expression_dict = {
                "neutral": None,
                "agree": "N3",
                "wonder": "N2",
                "shy": "N4",
                "happy": "N1",
                "sad": "N5",
            }

        available_hotkey_ids = (await self.send('HotkeysInCurrentModelRequest', None))['data']['availableHotkeys']
        for each_hotkey in available_hotkey_ids:
            name = each_hotkey['name'].lower()
            expression_dict[name] = each_hotkey['hotkeyID']

        try:
            hotkey_id = expression_dict[expression]
        except KeyError:
            print("Invalid expression")
            print(f'Available expressions: {expression_dict.keys()}')
            return expression_dict.keys()

        if hotkey_id is not None:
            msg = {
                "hotkeyID": hotkey_id,
            }
            result = await self.send("HotkeyTriggerRequest", msg)
            return result

if __name__ == "__main__":
    waifu = Char_control()
    print('------------------------------------')
    asyncio.run(waifu.express('happy'))


