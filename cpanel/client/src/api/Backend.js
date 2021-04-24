import queryString from 'query-string'
import config from '../config'

export const requestParams = httpMethod => ({
    method: httpMethod.toUpperCase(),
    mode: 'cors',
    cache: 'no-cache',
    credentials: 'same-origin',
    headers: {},
    redirect: 'follow',
    referrer: 'no-referrer'
});

export default class Backend {
    static request = async (method, params, httpMethod = 'GET') => {
        let url = config.api_basepath + method
        const reqParams = requestParams(httpMethod)

        if (httpMethod.toString().toUpperCase() !== 'GET') {
            if (!(params instanceof FormData)) {
                reqParams['headers']['Content-Type'] = 'application/json'
            }
            reqParams['body'] = params instanceof FormData ? params : JSON.stringify(params)
        } else {
            url += `?${queryString.stringify(params)}`;
        }

        try {
            let r = await fetch(url, reqParams);
            return await r.json();
        } catch (e) {
            console.log(e)
            return Promise.reject(e);
        }
    };
}