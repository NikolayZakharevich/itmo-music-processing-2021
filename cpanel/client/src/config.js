const configs = {
    local: {
        "api_basepath": 'https://sky4uk.xyz/api/'
    }
};

const returnConfig = configs[process.env.REACT_APP_CONFIG] || configs.local

export default returnConfig;