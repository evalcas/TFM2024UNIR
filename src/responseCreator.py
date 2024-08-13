from flask import jsonify

class ResponseCreator:
    @staticmethod
    def create_response(status, message, data=None):
        response = {
            "status": status,
            "message": message
        }
        if data is not None:
            response["data"] = data
        return jsonify(response)
