from flask import jsonify

class CustomExceptionHandler:
    @staticmethod
    def handle_global_exception(e):
        response = {
            "error": "Internal Server Error",
            "message": str(e)
        }
        return jsonify(response), 500

    @staticmethod
    def handle_bad_request(e):
        response = {
            "error": "Bad Request",
            "message": str(e)
        }
        return jsonify(response), 400

    @staticmethod
    def handle_not_found(e):
        response = {
            "error": "Not Found",
            "message": str(e)
        }
        return jsonify(response), 404
